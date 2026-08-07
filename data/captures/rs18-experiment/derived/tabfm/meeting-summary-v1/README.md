# TabFM soft-GMD meeting figure

`tabfm-soft-gmd-meeting` combines three views of the accepted RS18 dataset:

1. actual 200 Hz Sensor-X and Sensor-Y samples during the sequence-2 sync field,
   normalized only by the recorded prelaunch transmitter-off mean and RMS;
2. the Sensor-Y physical in-band SNR for the nine frames used in soft-GMD
   development and prospective confirmation;
3. hard frontend symbol errors and final payload outcomes for the frozen TabFM
   soft-GMD/list decoder.

Orange bars and the left side of the outcome matrix are method-development
frames. Blue bars and the right side are the prospectively selected confirmation
frames. Sequence 24 was already decodable before soft GMD; sequence 2 is the one
additional exploratory recovery. All four prospective candidates were rejected,
and later truth diagnostics confirmed that rejection prevented miscorrections.

Use the SVG for editing, the PDF for slides or printing, and the PNG for quick
sharing. `tabfm-soft-gmd-meeting.plot-data.csv` contains the plotted scalar data.
The figure is reproducible with
`experiments/tabfm_decoder/plot_meeting_summary.py`. All outputs have sibling
SHA-256 sidecars.
