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

## Post-build validation findings
- 45.0% of all (student, week) matrices are uniform (sparse/inactive week).
- Uniform rate differs by label: 51.5% for withdrawn students vs 44.4%
  for non-withdrawn — a real behavioral signal, not a bug.
- Label rate per week declines from 21.6% (week 0) to ~0% (week 35+),
  a survivorship artifact of post-withdrawal truncation (Decision #5).
  This must be reported per-week, not as a single pooled accuracy number.\

## Baseline representation decisions
- GAF/RP/MTF operate per-activity on the 7-day within-week click sequence
  (20 channels of length-7 series), analogous to ATI-v1's 20-activity
  structure, using the pyts library.
- MTF uses n_bins=4, strategy="uniform" (not the pyts default "quantile")
  because click sequences contain many tied/zero values that break
  quantile binning.
- Heatmap and Radar are implemented WITHOUT matplotlib for performance:
  matplotlib rendering at ~577k samples would be prohibitively slow.
  Heatmap: 20-dim weekly click vector reshaped into a fixed 4x5 grid.
  Radar: 20-vertex polygon rasterized via PIL.ImageDraw onto a 64x64 canvas.