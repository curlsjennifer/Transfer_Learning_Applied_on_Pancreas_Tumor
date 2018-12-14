# Data Arrangement

Content: Everything about data arrangement

## Notebook
`data_cleaning.ipynb`: move dicom with label from **raw_data** to **label_data**
`create_boxdata.ipynb`: create 3D numpy array
`data_sandbox.ipynb`: everything testing
`data_visualization.ipynb`: visualization the box data

## Library
### `arrangement.py`
Everything about file moving.
* move_labeldata_finecut
* move_labeldata_55cut
* move_nolabeldata
* sort_date
* sort_series

### `checking.py`
Everything about checking.
* check_dcm
* refine_dcm
* check_avphase

## TODO

### Black list format
All the list are in `../../doc/`
