# core/actions/ura/lobby.py
from __future__ import annotations


from core.actions.lobby import LobbyFlow
from core.settings import Settings
from core.controllers.base import IController
from core.perception.extractors.state import (
    extract_mood,
    extract_infirmary_on,
    extract_skill_points,
    extract_goal_text,
    extract_energy_pct,
    extract_turns,
)
from core.perception.yolo.interface import IDetector
from core.utils.logger import logger_uma
from core.utils.race_index import date_key_from_dateinfo
from core.utils.waiter import Waiter
from core.utils.yolo_objects import collect

from core.utils.date_uma import (
    is_summer_in_two_or_less_turns,
    date_is_confident,
    date_index,
)

class LobbyFlowURA(LobbyFlow):
    """
    Encapsulates all Lobby decisions & navigation.
    Composes RaceFlow and centralizes waits via a single Waiter.
    """

    def __init__(
        self,
        ctrl: IController,
        ocr,
        yolo_engine: IDetector,
        waiter: Waiter,
        *,
        minimum_skill_pts: int = 500,
        auto_rest_minimum: int = 24,
        prioritize_g1: bool = False,
        process_on_demand=True,
        interval_stats_refresh=1,
        max_critical_turn=8,
        plan_races={},
    ) -> None:
        super().__init__(
            ctrl,
            ocr,
            yolo_engine,
            waiter,
            minimum_skill_pts=minimum_skill_pts,
            auto_rest_minimum=auto_rest_minimum,
            prioritize_g1=prioritize_g1,
            process_on_demand=process_on_demand,
            interval_stats_refresh=interval_stats_refresh,
            max_critical_turn=max_critical_turn,
            plan_races=plan_races,
        )

    # --------------------------
    # Public entry point
    # --------------------------
    def process_turn(self):
        """
        Evaluate the Lobby and take the next action.
        Returns a short outcome string:
          - "RACED"          → we entered/finished a race flow
          - "INFIRMARY"      → we went to infirmary
          - "RESTED"         → we chose rest/recreation
          - "TO_TRAINING"    → we navigated to the training screen
          - "TRAINING_READY" → we're in training with tile already clicked (pre-check optimization)
          - "CONTINUE"       → we did a minor click or nothing

        optional extra message
        """
        img, dets = collect(
            self.yolo_engine,
            imgsz=self.waiter.cfg.imgsz,
            conf=self.waiter.cfg.conf,
            iou=self.waiter.cfg.iou,
            tag="lobby_state",
        )
        # Update PAL availability flag on-demand (centralized helper)
        self._update_pal_from_dets(dets)

        if not self.process_on_demand:
            self._update_state(
                img, dets
            )  # -> Very expensive, calculate as you need better

        # --- Critical goal logic & early racing opportunities ---

        if self.process_on_demand:
            self._process_date_info(img, dets)
            logger_uma.info(
                f"Date: {self.state.date_info} | raw: {self.state.career_date_raw}"
            )
            # Align PAL memory metadata with the current run/date
            self._refresh_pal_memory()

        if self.process_on_demand:
            self.state.energy = extract_energy_pct(img, dets)

        # --- Race planning (explicit list takes precedence; else G1 if available) ---
        self._plan_race_today()

        current_date_key = (
            date_key_from_dateinfo(self.state.date_info)
            if self.state and getattr(self.state, "date_info", None)
            else None
        )
        if (
            self._skip_race_once
            and self._skip_guard_key
            and current_date_key
            and current_date_key != self._skip_guard_key
        ):
            logger_uma.info(
                "[planned_race] skip guard released: %s -> %s",
                self._skip_guard_key,
                current_date_key,
            )
            self._skip_race_once = False
            self._skip_guard_key = None

        # If we have a planned race today, go race (subject to early-guard rules)
        if self.state.planned_race_name:
            # First-Junior-Day guard (no races available there)
            is_first_junior_date = (
                bool(self.state.date_info)
                and date_is_confident(self.state.date_info)
                and self.state.date_info.year_code == 1
                and self.state.date_info.month == 7
                and self.state.date_info.half == 1
            )
            guard_extra = {
                "first_junior": is_first_junior_date,
                "skip_guard": self._skip_race_once,
                "tentative": self.state.planned_race_tentative,
            }
            self._log_planned_race_decision(
                action="guard_evaluate",
                plan_name=self.state.planned_race_name,
                extra=guard_extra,
            )
            if not is_first_junior_date and not self._skip_race_once:
                if self.state.planned_race_tentative:
                    skip_training, skip_reason = self._should_skip_planned_race_for_training(img, dets)
                    if skip_training:
                        self._log_planned_race_decision(
                            action="precheck_skip",
                            plan_name=self.state.planned_race_name,
                            reason=skip_reason,
                            extra=guard_extra,
                        )
                        # Check if tile already clicked by pre-check optimization
                        if "[tile_clicked]" in skip_reason:
                            return "TRAINING_READY", skip_reason
                        # Otherwise navigate to training
                        if self._go_training_screen_from_lobby(img, dets):
                            return "TO_TRAINING", skip_reason
                        return "CONTINUE", skip_reason
                reason = f"Planned race: {self.state.planned_race_name}"
                self._log_planned_race_decision(
                    action="enter_race",
                    plan_name=self.state.planned_race_name,
                    reason="guard_passed",
                    extra=guard_extra,
                )
                self._skip_race_once = False
                return "TO_RACE", reason
            else:
                # --- OCR retry: re-screenshot and re-parse date before
                # accepting the suppression.  The first-junior-date guard
                # is particularly sensitive to OCR misreads on the date
                # pill; a fresh capture often resolves the error. --------
                if is_first_junior_date:
                    _MAX_DATE_RETRIES = 2
                    for _retry_i in range(1, _MAX_DATE_RETRIES + 1):
                        import time as _time
                        _time.sleep(0.3)
                        retry_img, retry_dets = collect(
                            self.yolo_engine,
                            imgsz=self.waiter.cfg.imgsz,
                            conf=self.waiter.cfg.conf,
                            iou=self.waiter.cfg.iou,
                            tag="lobby_date_retry",
                        )
                        self._process_date_info(retry_img, retry_dets)
                        logger_uma.info(
                            "[lobby] Date retry %d/%d: %s | raw: %s",
                            _retry_i, _MAX_DATE_RETRIES,
                            self.state.date_info,
                            self.state.career_date_raw,
                        )
                        # Re-evaluate the guard with the fresh date
                        is_first_junior_date = (
                            bool(self.state.date_info)
                            and date_is_confident(self.state.date_info)
                            and self.state.date_info.year_code == 1
                            and self.state.date_info.month == 7
                            and self.state.date_info.half == 1
                        )
                        if not is_first_junior_date:
                            logger_uma.info(
                                "[lobby] Date retry cleared first_junior_date guard → proceeding to race"
                            )
                            break

                    # If retry cleared the guard, go to race
                    if not is_first_junior_date and not self._skip_race_once:
                        reason = f"Planned race (after date retry): {self.state.planned_race_name}"
                        self._log_planned_race_decision(
                            action="enter_race_after_retry",
                            plan_name=self.state.planned_race_name,
                            reason="date_retry_cleared",
                        )
                        self._skip_race_once = False
                        return "TO_RACE", reason

                # Still suppressed after retries (or suppressed by skip_guard)
                suppression_reasons = []
                if is_first_junior_date:
                    suppression_reasons.append("first_junior_date")
                if self._skip_race_once:
                    suppression_reasons.append("skip_guard")
                self._log_planned_race_decision(
                    action="guard_suppressed",
                    plan_name=self.state.planned_race_name,
                    reason=",".join(suppression_reasons) or "unknown",
                    extra=guard_extra,
                )
                logger_uma.debug(
                    "[lobby] Planned race suppressed by first-junior-day/skip flag."
                )
            self._skip_race_once = False
            self._skip_guard_key = None

        if self.process_on_demand:
            self._process_turns_left(img, dets)

        if self.state.turn <= self.max_critical_turn:
            if not self._skip_race_once and self.state.energy is not None and self.state.energy > 2:
                # [Optimization] 10 steps for goal, or unknown turns or -1 turns, check goal
                outcome_bool, reason = self._maybe_do_goal_race(img, dets)
                if outcome_bool:
                    return "TO_RACE", reason
                # Check if goal race pre-check clicked tile (outcome_bool=False means skip race)
                elif not outcome_bool and "[tile_clicked]" in reason:
                    return "TRAINING_READY", reason

        # After special-case goal racing, clear the one-shot skip guard.
        self._skip_race_once = False

        if self.process_on_demand:
            self.state.infirmary_on = extract_infirmary_on(img, dets, threshold=0.60)

        # --- Infirmary handling (only outside summer) ---
        if self.state.infirmary_on and (self.state.is_summer is False):
            if self._precheck_allowed():
                best_sv, meta = self._peek_training_best_sv(img, dets, stay_if_above_threshold=True)
                if best_sv >= Settings.RACE_PRECHECK_SV:
                    logger_uma.info(
                        "[lobby] Infirmary pre-check skip: sv=%.2f threshold=%.2f meta=%s",
                        best_sv,
                        Settings.RACE_PRECHECK_SV,
                        meta,
                    )
                    # Tile already clicked, return TRAINING_READY to skip scan
                    if meta.get("tile_clicked"):
                        return "TRAINING_READY", f"Pre-check tile clicked sv={best_sv:.2f}"
                    # Already in training screen but not clicked
                    if meta.get("stayed_in_training"):
                        return "TO_TRAINING", f"Pre-check training sv={best_sv:.2f}"
                    # Fallback: navigate if peek didn't stay
                    if self._go_training_screen_from_lobby(img, dets):
                        return "TO_TRAINING", f"Pre-check training sv={best_sv:.2f}"
                    return "CONTINUE", "Pre-check training after infirmary skip"
            if self._go_infirmary():
                return "INFIRMARY", "Infirmary to remove blue condition"

        # --- Energy management (rest / prefer PAL recreation) ---
        if self.state.energy is not None:
            # Precompute PAL readiness and late-season guard (Y3-Sep-1 and later).
            mem = getattr(self, "pal_memory", None)
            pal_ready = (
                getattr(self.state, "pal_available", False)
                and mem is not None
                and mem.any_next_energy()
            )

            late_pal_window = False
            if self.state.date_info is not None:
                idx = date_index(self.state.date_info)
                if idx is not None and idx >= 64:  # Y3-Sep-1 using date_index()
                    late_pal_window = True

            if self.state.energy <= self.auto_rest_minimum:
                # Late-season: always use PAL recreation when possible to avoid losing trainings.
                if pal_ready and late_pal_window:
                    reason = (
                        "Auto-rest: late-season PAL recreation (Y3-Sep-1+; avoid lost trainings)"
                    )
                    if self._go_recreate(reason=reason):
                        return "RESTED", reason

                # Otherwise prefer PAL recreation if available and mood < GREAT
                mood_lbl, mood_score = (
                    self.state.mood
                    if isinstance(self.state.mood, tuple) and len(self.state.mood) == 2
                    else ("UNKNOWN", -1)
                )
                if mood_score < 0:
                    try:
                        self.state.mood = extract_mood(self.ocr, img, dets, conf_min=0.3)
                        _, mood_score = self.state.mood
                    except Exception:
                        mood_score = -1
                if (
                    mood_score >= 0
                    and mood_score < 5  # MOOD_MAP["GREAT"]
                    and pal_ready
                ):
                    reason = (
                        f"Auto-rest: prefer PAL recreation (min={self.auto_rest_minimum}; mood<Great)"
                    )
                    if self._go_recreate(reason=reason):
                        return "RESTED", reason

                reason = f"Energy too low, resting: auto_rest_minimum={self.auto_rest_minimum}"
                if self._go_rest(reason=reason):
                    return "RESTED", reason

            elif (
                self.state.energy <= 50
                and self.state.date_info
                and is_summer_in_two_or_less_turns(self.state.date_info)
            ):
                # Prefer PAL recreation when preparing for summer if available
                if pal_ready:
                    reason = "Preparing for summer: prefer PAL recreation"
                    if self._go_recreate(reason=reason):
                        return "RESTED", reason
                reason = "Resting to prepare for summer"
                if self._go_rest(reason=reason):
                    return "RESTED", reason

        if self.process_on_demand:
            self.state.mood = extract_mood(self.ocr, img, dets, conf_min=0.3)

        if self.state.mood[-1] < 0:
            logger_uma.warning("UNKNOWN mood!")
        # --- Mood (for training policy)---
        # Navigate to Training if nothing else

        if self._go_training_screen_from_lobby(img, dets):
            return "TO_TRAINING", "No critical stuff going to train"

        return "CONTINUE", "Unknown"

    def _update_state(self, img, dets) -> None:
        # Skill points, goal & energy
        self.state.skill_pts = extract_skill_points(self.ocr, img, dets)
        self.state.goal = extract_goal_text(self.ocr, img, dets)
        self.state.energy = extract_energy_pct(img, dets)

        self._update_stats(img, dets)
        # Turns & career date parsing
        self._process_turns_left(img, dets)
        self._process_date_info(img, dets)
        # Infirmary & mood
        self.state.infirmary_on = extract_infirmary_on(img, dets, threshold=0.60)
        self.state.mood = extract_mood(self.ocr, img, dets, conf_min=0.3)

    def _process_turns_left(self, img, dets):
        new_turn = extract_turns(self.ocr, img, dets)
        if new_turn != -1:
            ref_turn = self.last_turns_left_prediction or self.state.turn or -1
            diff = abs(ref_turn - new_turn)
            if diff < 5 or ref_turn == -1 or self.state.turn <= 2:
                # accept only if new change is not to big, or if last detected turn was near to finish turns left
                self.state.turn = new_turn
            elif self.state.turn > 1:
                logger_uma.debug(
                    f"Naive prediction. Last turn was: {self.state.turn}, so now it should be at most {self.state.turn - 1}"
                )
                self.state.turn -= 1
        elif self.state.turn > 0:
            # Naive prediction
            logger_uma.debug(
                f"Naive prediction. Last turn was: {self.state.turn}, so now it should be at most {self.state.turn - 1}"
            )
            self.state.turn -= 1
        self.last_turns_left_prediction = new_turn
