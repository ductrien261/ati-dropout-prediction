# Design Decisions Log

## Data source
- OULAD via Kaggle mirror `mexwell/open-university-learning-analytics`.
- 7 raw tables confirmed: studentInfo, studentRegistration, studentVle, vle, courses, assessments, studentAssessment.

## Preprocessing decisions

1. **Transition granularity: day-level, not session-level (30-min gap).**
   Reason: `studentVle.csv["date"]` is an integer day offset from course
   start - no hour/minute timestamp exists. Confirmed empirically:
   see `q4_daily_activity_variety.png` - 84% of student-days involve more
   than one activity type with no recorded ordering, so within-day
   sequence cannot be determined. ATI transitions are therefore defined
   between activity vectors of consecutive days (t, t+1), weighted by
   click counts, not between individual timestamped events.

2. **Duplicate rows in studentVle.csv.**
   73+ rows sharing the exact same (code_module, code_presentation,
   id_student, id_site, date) key were found with different sum_click
   values. These are summed via groupby before any further processing.

3. **Label definition: fixed per (student, module, presentation), not per week.**
   `final_result == "Withdrawn"` -> label = 1, else 0. This same label
   is used for every week of that student (early-prediction setup):
   given only week-w behavior, predict the eventual outcome.

4. **Week range: start from week 0, drop negative weeks.**
   Negative `date` values (pre-course-start interaction) are excluded.

5. **Post-withdrawal truncation.**
   Rows with `date > date_unregistration` are dropped - a withdrawn
   student's post-withdrawal weeks (naturally all-zero) are excluded
   from training data to avoid a trivial shortcut signal.   