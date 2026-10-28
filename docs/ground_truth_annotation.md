# Ground-truth annotation guide

The MOT metrics (IDSW, IDF1, MOTA, HOTA) need hand-checked boxes for every player in every frame of a few hard clips. The pipeline pre-fills them, so the job is **correcting**, not drawing from scratch.

## The clips

`python -m padel_reid gtclip` writes the clips to `outputs/gt_clips/`:

| Clip | Video frames | Length | What makes it hard |
|---|---|---|---|
| `far_crossing_0415` | 7650–9150 | 50 s | Longest far-side overlap (92 frames, one player behind the other), plus two more close encounters |
| `far_crossing_1140` | 20400–21750 | 45 s | Two back-to-back far-side crossings; near players stepping out of the bottom edge |
| `changeover_1540` | 27600–28950 | 45 s | Teams walk to the other end and pass each other |
| `near_exit_crossing_2010` | 36000–37700 | 57 s | Player 3 out of frame for 10.9 s; a near-side crossing |

Each clip folder contains:
- `clip.mp4`
- `clip.json`, the clip's start frame in the full video
- `annotations_for_cvat.zip`, the pre-annotation in MOT 1.1 format
- `gt/gt.txt`, the file to replace with the corrected annotations
- `gt/preannotation.txt`, an untouched copy of the draft

## Player identities (keep these IDs)

| ID | Team | Appearance |
|---|---|---|
| 1 | A | black shirt, khaki cap, red paddle |
| 2 | A | white/blond hair, black-and-white shirt, black paddle |
| 3 | B | grey shirt, white shoes, cap, red/dark paddle |
| 4 | B | grey shirt, black socks, green-trimmed shorts, white paddle |

Team A starts on the near half; the teams change ends at about 15:40 (inside `changeover_1540`).

## Steps in CVAT (app.cvat.ai or a local CVAT)

1. Create a task with the label **`player`** and upload `clip.mp4`.
2. **Actions → Upload annotations → format "MOT 1.1"**, and choose `annotations_for_cvat.zip`.
3. Go through the clip and fix every frame:
   - **Wrong ID:** use *merge* / *split* on tracks so that every player keeps their single ID (1–4) for the whole clip.
   - **Missing player** (visible but no box): add a keyframe. CVAT interpolates between keyframes, so one every 5–10 frames is enough while the player moves smoothly.
   - **Box on a non-player, or a duplicate box:** delete it.
   - **Player fully out of frame or fully hidden:** mark the track *outside* for those frames. Don't leave a box there.
   - **Partly visible** (cut by the bottom edge, or half hidden): box only the visible part.
4. **Actions → Export task dataset → "MOT 1.1"**, and replace `gt/gt.txt` with the exported `gt/gt.txt`.

## Scoring

```bash
python -m padel_reid evaluate --gt outputs/gt_clips/far_crossing_0415/gt/gt.txt
```

Every method is scored on the clip: ByteTrack, raw BoT-SORT, the Baek & Yun-style position lock, our pipeline without the side/team cost (ablation), and our proposed pipeline. Results are written to `outputs/metrics/metrics_<clip>.csv/json`.

The command refuses to run while `gt.txt` is still identical to `preannotation.txt`, because scoring the pipeline against its own output would be meaningless.
