# Capstone Project: Group 15, Domain 15 (Hospitality & Tourism)

**Team:** Shafi Shaik, Bhagyashree Patil, Kavita Rawat, Savithabharathi D
**Status:** Pre-submission draft v1, shared for instructor review (deadline 4 Nov 2026, tentative)

## Pre-submission proposal
- PDF (3 pages): [pre_submission/Group15_Domain15_PreSubmission_v1.pdf](pre_submission/Group15_Domain15_PreSubmission_v1.pdf)
- Editable source: `pre_submission/Group15_Domain15_PreSubmission_v1.docx`

## Business problem
HorizonStay Hotels sells through Direct, OTA, Corporate and Travel Agent channels.
Bookings are growing, but 27% of bookings never complete and revenue is uneven.

## Objectives
1. Primary: predict at booking time whether a reservation will complete, cancel or no-show.
2. Secondary: compare booked, realized and net revenue by channel, hotel and segment.
3. Stretch: find factors linked to repeat booking and guest satisfaction.

## Data
Three files provided for Domain 15, kept unchanged in `data/`:
- `Hospitality_Tourism.csv`: 17,661 bookings, 54 columns
- `Hospitality_Tourism_Room_Availability.csv`: 13,158 daily room-inventory rows
- `Hospitality_Tourism_Data_Dictionary.csv`: what each column means and when it becomes known

## Analysis so far
`code/Group15_Domain15_Analysis.ipynb` reproduces every number in the proposal and explains each
result in plain language. Charts are saved in `figures/`.

**To run it:** open the notebook from inside this repository, so that `code/` sits next to `data/`.
It needs Python 3 with `pandas`, `numpy`, `matplotlib` and `jupyter`
(`pip install pandas numpy matplotlib jupyter`). No model is trained yet.

## Repository structure
```
capstone_project/
├── README.md
├── code/            analysis notebook
├── data/            the 3 provided CSV files (do not edit)
├── docs/            problem statement from the course
├── figures/         charts produced by the notebook
└── pre_submission/  proposal (PDF + editable .docx)
```
Added later: `reports/` (final technical report), `presentation/`.

## Ownership
| Area | Owner |
|---|---|
| Business problem and objectives | |
| Challenges and data audit | |
| Literature review | |
| Methodology, costs, timeline | |

## Working rules
- Commit under your own name; one change per commit with a clear message.
- Another member reviews each change before it is merged.