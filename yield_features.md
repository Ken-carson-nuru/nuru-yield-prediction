**Imports**


```python
#!pip install pydrive
#!pip install gdown pandas
```


```python
#Importing libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import io
import requests
import json
warnings.filterwarnings('ignore')


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import io
import gdown
import zipfile
import os

```

**Authenticate google drive**



```python
# gauth = GoogleAuth()
# gauth.LocalWebserverAuth()  # or gauth.CommandLineAuth() depending on environment
# drive = GoogleDrive(gauth)

```


```python
# Replace with your actual sheet-ID and sheet GID
sheet_id = "1nBiruECDXrBd7BgX4gStJXWKFYBNU_IErU5nP7sRih0"
gid = "133044706"

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
df = pd.read_csv(url)

df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boxes_box1_gps</th>
      <th>boxes_crop</th>
      <th>boxes_planting_date</th>
      <th>boxes_expected_harvest_date</th>
      <th>country</th>
      <th>boxes_farmer_district</th>
      <th>wet_completed_time</th>
      <th>wet_kgs_crop_box1</th>
      <th>wet_kgs_crop_box2</th>
      <th>dry_box1_dry_weight</th>
      <th>dry_box2_dry_weight</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Altitude</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.499127 37.6122529 1252.08 7.62</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-03-20</td>
      <td>4.910</td>
      <td>8.810</td>
      <td>2.600</td>
      <td>3.395</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>1252.08</td>
      <td>7.62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.4754511 37.6402894 1234.5 9.73</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-03-03</td>
      <td>12.170</td>
      <td>10.520</td>
      <td>3.195</td>
      <td>4.415</td>
      <td>-0.475451</td>
      <td>37.640289</td>
      <td>1234.50</td>
      <td>9.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.5351206 37.6086503 1226.4 4.5</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-02-23</td>
      <td>1.245</td>
      <td>1.575</td>
      <td>0.495</td>
      <td>0.535</td>
      <td>-0.535121</td>
      <td>37.608650</td>
      <td>1226.40</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.3971582 37.5027862 1839.7 5.25</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-04-06</td>
      <td>3.700</td>
      <td>3.925</td>
      <td>3.555</td>
      <td>3.470</td>
      <td>-0.397158</td>
      <td>37.502786</td>
      <td>1839.70</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.4509697 37.6600589 1234.53 8.0</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-03-25</td>
      <td>4.645</td>
      <td>4.815</td>
      <td>3.605</td>
      <td>3.695</td>
      <td>-0.450970</td>
      <td>37.660059</td>
      <td>1234.53</td>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Preview

print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)
```

    
    Columns: ['boxes_box1_gps', 'boxes_crop', 'boxes_planting_date', 'boxes_expected_harvest_date', 'country', 'boxes_farmer_district', 'wet_completed_time', 'wet_kgs_crop_box1', 'wet_kgs_crop_box2', 'dry_box1_dry_weight', 'dry_box2_dry_weight', 'Latitude', 'Longitude', 'Altitude', 'Accuracy']
    
    Shape: (428, 15)
    


```python
# Add a unique farm_id starting from 1
df.insert(0, 'Plot_No', range(1, len(df) + 1))
```

**Clean and Rename columns**


```python
# Drop the unwanted column
df = df.drop(columns=['boxes_box1_gps'], errors='ignore')

# Rename specific columns
df = df.rename(columns={
    'boxes_crop': 'crop_type',
    'boxes_planting_date': 'planting_date',
    'boxes_expected_harvest_date': 'expected_harvest_date',
    'boxes_farmer_district': 'county'
})


# Convert planting and harvest dates to datetime
df['planting_date'] = pd.to_datetime(df['planting_date'], errors='coerce')
df['expected_harvest_date'] = pd.to_datetime(df['expected_harvest_date'], errors='coerce')
df['wet_completed_time'] = pd.to_datetime(df['wet_completed_time'], errors='coerce')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>wet_completed_time</th>
      <th>wet_kgs_crop_box1</th>
      <th>wet_kgs_crop_box2</th>
      <th>dry_box1_dry_weight</th>
      <th>dry_box2_dry_weight</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Altitude</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-03-20</td>
      <td>4.910</td>
      <td>8.810</td>
      <td>2.600</td>
      <td>3.395</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>1252.08</td>
      <td>7.62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-03-03</td>
      <td>12.170</td>
      <td>10.520</td>
      <td>3.195</td>
      <td>4.415</td>
      <td>-0.475451</td>
      <td>37.640289</td>
      <td>1234.50</td>
      <td>9.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-02-23</td>
      <td>1.245</td>
      <td>1.575</td>
      <td>0.495</td>
      <td>0.535</td>
      <td>-0.535121</td>
      <td>37.608650</td>
      <td>1226.40</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-04-06</td>
      <td>3.700</td>
      <td>3.925</td>
      <td>3.555</td>
      <td>3.470</td>
      <td>-0.397158</td>
      <td>37.502786</td>
      <td>1839.70</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>2022-03-25</td>
      <td>4.645</td>
      <td>4.815</td>
      <td>3.605</td>
      <td>3.695</td>
      <td>-0.450970</td>
      <td>37.660059</td>
      <td>1234.53</td>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
</div>



**Order columns**


```python
# Define the preferred column order
preferred_order = ['Plot_No', 'Longitude','Latitude', 'Altitude','crop_type', 'planting_date','wet_completed_time', 'expected_harvest_date']

# Add remaining columns to the order dynamically
remaining_cols = [col for col in df.columns if col not in preferred_order]
df = df[preferred_order + remaining_cols]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>wet_completed_time</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>wet_kgs_crop_box1</th>
      <th>wet_kgs_crop_box2</th>
      <th>dry_box1_dry_weight</th>
      <th>dry_box2_dry_weight</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>2022-03-20</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.910</td>
      <td>8.810</td>
      <td>2.600</td>
      <td>3.395</td>
      <td>7.62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>2022-03-03</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>12.170</td>
      <td>10.520</td>
      <td>3.195</td>
      <td>4.415</td>
      <td>9.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>2022-02-23</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>1.245</td>
      <td>1.575</td>
      <td>0.495</td>
      <td>0.535</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>2022-04-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>3.700</td>
      <td>3.925</td>
      <td>3.555</td>
      <td>3.470</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>2022-03-25</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.645</td>
      <td>4.815</td>
      <td>3.605</td>
      <td>3.695</td>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
</div>



**Compute average wet and dry harvests**


```python
# Ensure numeric conversion to avoid string issues
df['wet_kgs_crop_box1'] = pd.to_numeric(df['wet_kgs_crop_box1'], errors='coerce')
df['wet_kgs_crop_box2'] = pd.to_numeric(df['wet_kgs_crop_box2'], errors='coerce')
df['dry_box1_dry_weight'] = pd.to_numeric(df['dry_box1_dry_weight'], errors='coerce')
df['dry_box2_dry_weight'] = pd.to_numeric(df['dry_box2_dry_weight'], errors='coerce')
```


```python
# Compute averages
# Each box = 40 m² → total of 80 m² for two boxes
box_area = 40
total_area = 2 * box_area  # 80 m²
m2_per_hectare = 10000

# Ensure numeric conversion to avoid issues
for col in ['wet_kgs_crop_box1', 'wet_kgs_crop_box2', 'dry_box1_dry_weight', 'dry_box2_dry_weight']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute total yield and convert to per hectare
df['wet_harvest'] = ((df['wet_kgs_crop_box1'] + df['wet_kgs_crop_box2']) / total_area) * m2_per_hectare
df['dry_harvest'] = ((df['dry_box1_dry_weight'] + df['dry_box2_dry_weight']) / total_area) * m2_per_hectare

# Round for clarity
df['wet_harvest_kg/ha'] = df['wet_harvest'].round(2)
df['dry_harvest_kg/ha'] = df['dry_harvest'].round(2)

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>wet_completed_time</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>wet_kgs_crop_box1</th>
      <th>wet_kgs_crop_box2</th>
      <th>dry_box1_dry_weight</th>
      <th>dry_box2_dry_weight</th>
      <th>Accuracy</th>
      <th>wet_harvest</th>
      <th>dry_harvest</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>2022-03-20</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.910</td>
      <td>8.810</td>
      <td>2.600</td>
      <td>3.395</td>
      <td>7.62</td>
      <td>1715.000</td>
      <td>749.375</td>
      <td>1715.00</td>
      <td>749.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>2022-03-03</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>12.170</td>
      <td>10.520</td>
      <td>3.195</td>
      <td>4.415</td>
      <td>9.73</td>
      <td>2836.250</td>
      <td>951.250</td>
      <td>2836.25</td>
      <td>951.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>2022-02-23</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>1.245</td>
      <td>1.575</td>
      <td>0.495</td>
      <td>0.535</td>
      <td>4.50</td>
      <td>352.500</td>
      <td>128.750</td>
      <td>352.50</td>
      <td>128.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>2022-04-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>3.700</td>
      <td>3.925</td>
      <td>3.555</td>
      <td>3.470</td>
      <td>5.25</td>
      <td>953.125</td>
      <td>878.125</td>
      <td>953.12</td>
      <td>878.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>2022-03-25</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.645</td>
      <td>4.815</td>
      <td>3.605</td>
      <td>3.695</td>
      <td>8.00</td>
      <td>1182.500</td>
      <td>912.500</td>
      <td>1182.50</td>
      <td>912.50</td>
    </tr>
  </tbody>
</table>
</div>



**Drop individual box columns now that averages are computed**


```python

df = df.drop(columns=[
    'wet_kgs_crop_box1',
    'wet_kgs_crop_box2',
    'dry_box1_dry_weight',
    'dry_box2_dry_weight'
], errors='ignore')

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>wet_completed_time</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>Accuracy</th>
      <th>wet_harvest</th>
      <th>dry_harvest</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>2022-03-20</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>7.62</td>
      <td>1715.000</td>
      <td>749.375</td>
      <td>1715.00</td>
      <td>749.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>2022-03-03</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>9.73</td>
      <td>2836.250</td>
      <td>951.250</td>
      <td>2836.25</td>
      <td>951.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>2022-02-23</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.50</td>
      <td>352.500</td>
      <td>128.750</td>
      <td>352.50</td>
      <td>128.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>2022-04-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>5.25</td>
      <td>953.125</td>
      <td>878.125</td>
      <td>953.12</td>
      <td>878.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>2022-03-25</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>8.00</td>
      <td>1182.500</td>
      <td>912.500</td>
      <td>1182.50</td>
      <td>912.50</td>
    </tr>
  </tbody>
</table>
</div>



**Extracting Planting Season**


```python
# Convert planting_date to datetime
df["planting_date"] = pd.to_datetime(df["planting_date"])
```


```python
# Define function to determine season
def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return "Long Rains"
    elif month in [10, 11, 12]:
        return "Short Rains"
    else:
        return "Off Season"
```


```python
# Apply function to create 'season' column
df["season"] = df["planting_date"].apply(get_season)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>wet_completed_time</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>Accuracy</th>
      <th>wet_harvest</th>
      <th>dry_harvest</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>2022-03-20</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>7.62</td>
      <td>1715.000</td>
      <td>749.375</td>
      <td>1715.00</td>
      <td>749.38</td>
      <td>Short Rains</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>2022-03-03</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>9.73</td>
      <td>2836.250</td>
      <td>951.250</td>
      <td>2836.25</td>
      <td>951.25</td>
      <td>Short Rains</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>2022-02-23</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.50</td>
      <td>352.500</td>
      <td>128.750</td>
      <td>352.50</td>
      <td>128.75</td>
      <td>Short Rains</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>2022-04-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>5.25</td>
      <td>953.125</td>
      <td>878.125</td>
      <td>953.12</td>
      <td>878.12</td>
      <td>Short Rains</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>2022-03-25</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>8.00</td>
      <td>1182.500</td>
      <td>912.500</td>
      <td>1182.50</td>
      <td>912.50</td>
      <td>Short Rains</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reorder columns — place 'season' right after 'planting_date'
cols = df.columns.tolist()
insert_pos = cols.index("planting_date") + 1
cols.insert(insert_pos, cols.pop(cols.index("season")))
df = df[cols]

# Display the DataFrame
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>season</th>
      <th>wet_completed_time</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>Accuracy</th>
      <th>wet_harvest</th>
      <th>dry_harvest</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>Short Rains</td>
      <td>2022-03-20</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>7.62</td>
      <td>1715.000</td>
      <td>749.375</td>
      <td>1715.00</td>
      <td>749.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>Short Rains</td>
      <td>2022-03-03</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>9.73</td>
      <td>2836.250</td>
      <td>951.250</td>
      <td>2836.25</td>
      <td>951.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>Short Rains</td>
      <td>2022-02-23</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.50</td>
      <td>352.500</td>
      <td>128.750</td>
      <td>352.50</td>
      <td>128.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>Short Rains</td>
      <td>2022-04-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>5.25</td>
      <td>953.125</td>
      <td>878.125</td>
      <td>953.12</td>
      <td>878.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>Short Rains</td>
      <td>2022-03-25</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>8.00</td>
      <td>1182.500</td>
      <td>912.500</td>
      <td>1182.50</td>
      <td>912.50</td>
    </tr>
  </tbody>
</table>
</div>



**SAve a few columns to be used in Visual crossing for planting date**


```python
# Assuming df already exists and has 'season' column
selected_columns = [
    "Plot_No",
    "Longitude",
    "Latitude",
    "Altitude",
    "crop_type",
    "planting_date",
    "season",
    "country",
    "county"
]

# Select only the specified columns
df_selected = df[selected_columns]

# Save to CSV
output_path = "selected_farm_data.csv"
df_selected.to_csv(output_path, index=False)
```


```python
df_selected.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>season</th>
      <th>country</th>
      <th>county</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>Short Rains</td>
      <td>Kenya</td>
      <td>Embu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>Short Rains</td>
      <td>Kenya</td>
      <td>Embu</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>Short Rains</td>
      <td>Kenya</td>
      <td>Embu</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>Short Rains</td>
      <td>Kenya</td>
      <td>Embu</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>Short Rains</td>
      <td>Kenya</td>
      <td>Embu</td>
    </tr>
  </tbody>
</table>
</div>



## **Weather Data Extraction**


```python
import time
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
```

**Visual Crossing API KEY**


```python
API_KEY = "HV8ZYXKJHRFAHLLBRY6EP87PQ"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1nBiruECDXrBd7BgX4gStJXWKFYBNU_IErU5nP7sRih0/export?format=csv&gid=133044706"
```


```python
# Retry settings
MAX_RETRIES = 5
BACKOFF_FACTOR = 5  
```


```python
df.columns = df.columns.str.strip().str.lower()
df = df.dropna(subset=['latitude', 'longitude', 'planting_date', 'wet_completed_time'])
df['planting_date'] = pd.to_datetime(df['planting_date'], errors='coerce')
df['wet_completed_time'] = pd.to_datetime(df['wet_completed_time'], errors='coerce')
print(f"✅ Loaded {len(df)} farms")
```

    ✅ Loaded 428 farms
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plot_no</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>altitude</th>
      <th>crop_type</th>
      <th>planting_date</th>
      <th>season</th>
      <th>wet_completed_time</th>
      <th>expected_harvest_date</th>
      <th>country</th>
      <th>county</th>
      <th>accuracy</th>
      <th>wet_harvest</th>
      <th>dry_harvest</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37.612253</td>
      <td>-0.499127</td>
      <td>1252.08</td>
      <td>maize</td>
      <td>2021-10-06</td>
      <td>Short Rains</td>
      <td>2022-03-20</td>
      <td>2022-03-25</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>7.62</td>
      <td>1715.000</td>
      <td>749.375</td>
      <td>1715.00</td>
      <td>749.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37.640289</td>
      <td>-0.475451</td>
      <td>1234.50</td>
      <td>maize</td>
      <td>2021-11-10</td>
      <td>Short Rains</td>
      <td>2022-03-03</td>
      <td>2022-03-12</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>9.73</td>
      <td>2836.250</td>
      <td>951.250</td>
      <td>2836.25</td>
      <td>951.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>37.608650</td>
      <td>-0.535121</td>
      <td>1226.40</td>
      <td>maize</td>
      <td>2021-12-02</td>
      <td>Short Rains</td>
      <td>2022-02-23</td>
      <td>2022-02-26</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>4.50</td>
      <td>352.500</td>
      <td>128.750</td>
      <td>352.50</td>
      <td>128.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37.502786</td>
      <td>-0.397158</td>
      <td>1839.70</td>
      <td>maize</td>
      <td>2021-11-06</td>
      <td>Short Rains</td>
      <td>2022-04-06</td>
      <td>2022-04-06</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>5.25</td>
      <td>953.125</td>
      <td>878.125</td>
      <td>953.12</td>
      <td>878.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37.660059</td>
      <td>-0.450970</td>
      <td>1234.53</td>
      <td>maize</td>
      <td>2021-11-14</td>
      <td>Short Rains</td>
      <td>2022-03-25</td>
      <td>2022-04-01</td>
      <td>Kenya</td>
      <td>Embu</td>
      <td>8.00</td>
      <td>1182.500</td>
      <td>912.500</td>
      <td>1182.50</td>
      <td>912.50</td>
    </tr>
  </tbody>
</table>
</div>




```python

def fetch_weather(row):
    plot_no = row['farm_id'] if 'farm_id' in row else row['plot_no']
    lat, lon = row['latitude'], row['longitude']
    start_date = pd.to_datetime(row['planting_date']).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(row['wet_completed_time']).strftime('%Y-%m-%d')

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
        f"timeline/{lat},{lon}/{start_date}/{end_date}"
        f"?unitGroup=metric&include=days&key={API_KEY}&contentType=json"
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()

                return [
                    {
                        "Plot_No": plot_no,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Date": pd.to_datetime(day["datetime"]),
                        "Max_Temp_C": day.get("tempmax"),
                        "Min_Temp_C": day.get("tempmin"),
                        "Mean_Temp_C": day.get("temp"),
                        "Precip_mm": day.get("precip", 0),
                        "Humidity_%": day.get("humidity"),
                        "SolarEnergy_MJm2": day.get("solarenergy"),
                        "Evapotranspiration_mm": day.get("et"),
                        "WindSpeed_kmh": day.get("windspeed"),
                        "CloudCover_%": day.get("cloudcover"),
                    }
                    for day in data.get("days", [])
                ]

            elif resp.status_code == 429:
                wait_time = BACKOFF_FACTOR * (2 ** retries)
                print(f"⚠️ {plot_no} — Rate limit hit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1

            else:
                print(f"❌ {plot_no} — Error {resp.status_code}: {resp.text}")
                return []

        except requests.exceptions.RequestException as e:
            wait_time = BACKOFF_FACTOR * (2 ** retries)
            print(f"⚠️ {plot_no} — Network error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            retries += 1

    print(f"❌ {plot_no} — Max retries reached. Skipping.")
    return []

```

**EXECUTION BLOCK**


```python
def fetch_weather(row):
    plot_no = row['farm_id'] if 'farm_id' in row else row['plot_no']
    lat, lon = row['latitude'], row['longitude']
    planting_date = pd.to_datetime(row['planting_date'])
    start_date = planting_date.strftime('%Y-%m-%d')
    end_date = pd.to_datetime(row['wet_completed_time']).strftime('%Y-%m-%d')

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
        f"timeline/{lat},{lon}/{start_date}/{end_date}"
        f"?unitGroup=metric&include=days&key={API_KEY}&contentType=json"
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()

                return [
                    {
                        "Plot_No": plot_no,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Planting_Date": planting_date.strftime('%Y-%m-%d'),
                        "Date": pd.to_datetime(day["datetime"]),
                        "Max_Temp_C": day.get("tempmax"),
                        "Min_Temp_C": day.get("tempmin"),
                        "Mean_Temp_C": day.get("temp"),
                        "Precip_mm": day.get("precip", 0),
                        "Humidity_%": day.get("humidity"),
                        "SolarEnergy_MJm2": day.get("solarenergy"),
                        "Evapotranspiration_mm": day.get("et"),
                        "WindSpeed_kmh": day.get("windspeed"),
                        "CloudCover_%": day.get("cloudcover"),
                    }
                    for day in data.get("days", [])
                ]

            elif resp.status_code == 429:
                wait_time = BACKOFF_FACTOR * (2 ** retries)
                print(f"⚠️ {plot_no} — Rate limit hit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1

            else:
                print(f"❌ {plot_no} — Error {resp.status_code}: {resp.text}")
                return []

        except requests.exceptions.RequestException as e:
            wait_time = BACKOFF_FACTOR * (2 ** retries)
            print(f"⚠️ {plot_no} — Network error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            retries += 1

    print(f"❌ {plot_no} — Max retries reached. Skipping.")
    return []


# ------------------------------------------
# EXECUTION BLOCK
# ------------------------------------------
print("🚀 Fetching weather data from Visual Crossing...")
records = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(fetch_weather, row) for _, row in df.iterrows()]
    for future in as_completed(futures):
        records.extend(future.result())

weather_df = pd.DataFrame(records)
print(f"🎯 Done! Retrieved {len(weather_df)} weather records across {weather_df['Plot_No'].nunique()} plots.")

weather_df.head()
```

    🚀 Fetching weather data from Visual Crossing...
    ⚠️ 4 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 5 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 3 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 1 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 2 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 39 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 71 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 66 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 76 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 105 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 139 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 140 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 144 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 195 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 199 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 200 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 216 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 249 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 250 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 274 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 275 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 299 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 340 — Rate limit hit (429). Retrying in 5s...
    ⚠️ 411 — Rate limit hit (429). Retrying in 5s...
    🎯 Done! Retrieved 64481 weather records across 428 plots.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Planting_Date</th>
      <th>Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>Evapotranspiration_mm</th>
      <th>WindSpeed_kmh</th>
      <th>CloudCover_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>-0.54792</td>
      <td>37.484666</td>
      <td>2021-10-21</td>
      <td>2021-10-21</td>
      <td>23.3</td>
      <td>16.0</td>
      <td>19.7</td>
      <td>8.092</td>
      <td>76.6</td>
      <td>15.3</td>
      <td>None</td>
      <td>11.0</td>
      <td>83.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>-0.54792</td>
      <td>37.484666</td>
      <td>2021-10-21</td>
      <td>2021-10-22</td>
      <td>26.6</td>
      <td>16.6</td>
      <td>21.2</td>
      <td>1.500</td>
      <td>70.4</td>
      <td>18.3</td>
      <td>None</td>
      <td>9.3</td>
      <td>65.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>-0.54792</td>
      <td>37.484666</td>
      <td>2021-10-21</td>
      <td>2021-10-23</td>
      <td>27.0</td>
      <td>17.3</td>
      <td>21.9</td>
      <td>1.700</td>
      <td>66.4</td>
      <td>20.5</td>
      <td>None</td>
      <td>14.5</td>
      <td>43.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>-0.54792</td>
      <td>37.484666</td>
      <td>2021-10-21</td>
      <td>2021-10-24</td>
      <td>28.6</td>
      <td>14.6</td>
      <td>22.3</td>
      <td>0.000</td>
      <td>61.3</td>
      <td>24.7</td>
      <td>None</td>
      <td>14.6</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>-0.54792</td>
      <td>37.484666</td>
      <td>2021-10-21</td>
      <td>2021-10-25</td>
      <td>28.5</td>
      <td>16.1</td>
      <td>22.3</td>
      <td>0.700</td>
      <td>64.3</td>
      <td>23.6</td>
      <td>None</td>
      <td>11.5</td>
      <td>54.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 🧩 Number of unique plots after Visual Crossing extraction
unique_plots = weather_df['Plot_No'].nunique()
unique_plots
```




    428




```python
# Count missing values per column
missing_summary = weather_df.isna().sum().reset_index()
missing_summary.columns = ['Column', 'Missing_Count']
missing_summary['Missing_%'] = (missing_summary['Missing_Count'] / len(weather_df)) * 100
```


```python
# Display nicely sorted output
missing_summary = missing_summary.sort_values(by='Missing_%', ascending=False)
missing_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Missing_Count</th>
      <th>Missing_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>Evapotranspiration_mm</td>
      <td>64481</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Plot_No</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Latitude</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Longitude</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Planting_Date</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Date</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Max_Temp_C</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Min_Temp_C</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mean_Temp_C</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Precip_mm</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Humidity_%</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SolarEnergy_MJm2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>WindSpeed_kmh</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CloudCover_%</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# **Determining Crop Stages**

**Define the GDD calculation function**


```python

def daily_gdd(max_temp, min_temp, base_temp=10):
    # Convert temperatures to float in case they are strings
    try:
        max_temp = float(max_temp)
        min_temp = float(min_temp)
    except ValueError:
        raise ValueError(f"Invalid temperature value: max_temp={max_temp}, min_temp={min_temp}")
    adjusted_avg_temp = (max_temp + min_temp) / 2 
    
    # Calculate the Growing Degree Days (GDD) for a single day.
    return max(adjusted_avg_temp - base_temp, 0)
```

# **Compute the cumulative GDD for each day starting from a specified date**


```python
def compute_cumulative_gdd(df):
    # Convert 'Date' and 'planting_date' columns to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
    df['Planting_Date'] = pd.to_datetime(df['Planting_Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
    
    # Initialize an empty DataFrame to store results
    df_filtered_all = pd.DataFrame()
    
    # Group the DataFrame by Plot_No
    for plot_no, plot_data in df.groupby('Plot_No'):
        # Get the planting date for the current plot
        planting_date = plot_data['Planting_Date'].iloc[0]
        
        # Filter the DataFrame for dates from the planting date onwards
        df_filtered = plot_data[plot_data['Date'] >= planting_date].copy()
        
        # Calculate the daily GDD for each row and add it as a new column 'GDD'
        df_filtered['GDD'] = df_filtered.apply(lambda row: daily_gdd(row['Max_Temp_C'], row['Min_Temp_C']), axis=1)
        
        # Compute cumulative GDD for the current plot
        df_filtered['Cumulative_GDD'] = df_filtered['GDD'].cumsum()
        
        # Append the filtered and processed data to the result DataFrame
        df_filtered_all = pd.concat([df_filtered_all, df_filtered])

    # Save the DataFrame with cumulative GDD to a new CSV file
    output_csv_file = 'Cumulative_GDD3.csv'
    df_filtered_all.to_csv(output_csv_file, index=False)
   
    return df_filtered_all

```


```python

df_GDD=compute_cumulative_gdd(weather_df)
df_GDD.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Planting_Date</th>
      <th>Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>Evapotranspiration_mm</th>
      <th>WindSpeed_kmh</th>
      <th>CloudCover_%</th>
      <th>GDD</th>
      <th>Cumulative_GDD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2399</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-06</td>
      <td>2021-10-06</td>
      <td>26.9</td>
      <td>12.0</td>
      <td>20.9</td>
      <td>1.5</td>
      <td>61.9</td>
      <td>24.8</td>
      <td>None</td>
      <td>15.7</td>
      <td>48.4</td>
      <td>9.45</td>
      <td>9.45</td>
    </tr>
    <tr>
      <th>2400</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-06</td>
      <td>2021-10-07</td>
      <td>25.5</td>
      <td>15.7</td>
      <td>20.2</td>
      <td>5.2</td>
      <td>67.4</td>
      <td>23.2</td>
      <td>None</td>
      <td>16.2</td>
      <td>70.1</td>
      <td>10.60</td>
      <td>20.05</td>
    </tr>
    <tr>
      <th>2401</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-06</td>
      <td>2021-10-08</td>
      <td>23.3</td>
      <td>16.4</td>
      <td>20.0</td>
      <td>14.0</td>
      <td>74.5</td>
      <td>20.5</td>
      <td>None</td>
      <td>16.6</td>
      <td>68.4</td>
      <td>9.85</td>
      <td>29.90</td>
    </tr>
    <tr>
      <th>2402</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-06</td>
      <td>2021-10-09</td>
      <td>24.3</td>
      <td>16.6</td>
      <td>20.6</td>
      <td>1.2</td>
      <td>70.8</td>
      <td>12.4</td>
      <td>None</td>
      <td>11.2</td>
      <td>82.6</td>
      <td>10.45</td>
      <td>40.35</td>
    </tr>
    <tr>
      <th>2403</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-06</td>
      <td>2021-10-10</td>
      <td>26.4</td>
      <td>17.5</td>
      <td>21.3</td>
      <td>0.7</td>
      <td>71.6</td>
      <td>16.7</td>
      <td>None</td>
      <td>17.9</td>
      <td>81.9</td>
      <td>11.95</td>
      <td>52.30</td>
    </tr>
  </tbody>
</table>
</div>




```python
output="weather_df.csv"
weather_df.to_csv(output,index=False)
```


```python
weather_df.columns
```




    Index(['Plot_No', 'Latitude', 'Longitude', 'Planting_Date', 'Date',
           'Max_Temp_C', 'Min_Temp_C', 'Mean_Temp_C', 'Precip_mm', 'Humidity_%',
           'SolarEnergy_MJm2', 'Evapotranspiration_mm', 'WindSpeed_kmh',
           'CloudCover_%'],
          dtype='object')




```python
merged_df=pd.read_csv('merged_df.csv')
merged_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Data_Planting_Date</th>
      <th>Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>Evapotranspiration_mm</th>
      <th>WindSpeed_kmh</th>
      <th>CloudCover_%</th>
      <th>Planting_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>-0.535121</td>
      <td>37.60865</td>
      <td>12/2/2021</td>
      <td>12/2/2021</td>
      <td>21.3</td>
      <td>15.7</td>
      <td>18.9</td>
      <td>5.596</td>
      <td>86.2</td>
      <td>13.5</td>
      <td>NaN</td>
      <td>6.5</td>
      <td>81.0</td>
      <td>10/7/2021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>-0.535121</td>
      <td>37.60865</td>
      <td>12/2/2021</td>
      <td>12/3/2021</td>
      <td>21.3</td>
      <td>16.4</td>
      <td>18.3</td>
      <td>35.309</td>
      <td>89.0</td>
      <td>9.7</td>
      <td>NaN</td>
      <td>8.4</td>
      <td>79.0</td>
      <td>10/7/2021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.535121</td>
      <td>37.60865</td>
      <td>12/2/2021</td>
      <td>12/4/2021</td>
      <td>23.6</td>
      <td>15.9</td>
      <td>19.3</td>
      <td>16.566</td>
      <td>85.1</td>
      <td>16.6</td>
      <td>NaN</td>
      <td>11.9</td>
      <td>64.2</td>
      <td>10/7/2021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.535121</td>
      <td>37.60865</td>
      <td>12/2/2021</td>
      <td>12/5/2021</td>
      <td>23.3</td>
      <td>16.4</td>
      <td>19.7</td>
      <td>7.245</td>
      <td>80.9</td>
      <td>18.4</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>55.7</td>
      <td>10/7/2021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>-0.535121</td>
      <td>37.60865</td>
      <td>12/2/2021</td>
      <td>12/6/2021</td>
      <td>24.8</td>
      <td>15.1</td>
      <td>19.8</td>
      <td>11.766</td>
      <td>79.3</td>
      <td>20.5</td>
      <td>NaN</td>
      <td>10.8</td>
      <td>70.2</td>
      <td>10/7/2021</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd

def determine_corn_crop_stages_with_ideal_gdd(df):
    # Convert 'Date' and 'planting_date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df['Planting_Date'] = pd.to_datetime(df['Planting_Date'])

    # Define crop stages with their GDD thresholds
    crop_stages = {
        "Emergence (VE)": 100,
        "V2 (Two leaves with collars visible)": 250,
        "V6 (Six leaves with collars visible)": 475,
        "VT (Tasseling)": 900,
        "R1 (Silking)": 1100,
        "R3 (Milk)": 1400,
        "R5 (Dent)": 1925,
        "R6 (Maturity)": 2450
    }

    # Initialize a list to store the rows for the dataframe
    crop_stage_data = []

    # Group DataFrame by Plot_No (assuming each plot has unique coordinates and planting date)
    for plot_no, plot_data in df.groupby('Plot_No'):
        # Get the planting date for the current plot
        planting_date = plot_data['Planting_Date'].iloc[0]
        Latitude=plot_data['Latitude'].iloc[1]
        Longitude=plot_data['Longitude'].iloc[2]

        # Compute cumulative GDD from the planting date
        df_gdd = compute_cumulative_gdd(plot_data)

        # For each crop stage, determine the date and GDD
        for stage, gdd_threshold in crop_stages.items():
            stage_date = df_gdd[df_gdd['Cumulative_GDD'] >= gdd_threshold].head(1)['Date']
            if not stage_date.empty:
                stage_date_value = stage_date.values[0]
                stage_gdd_value = df_gdd[df_gdd['Date'] == stage_date_value]['Cumulative_GDD'].values[0]
                
                # Append the data to the list
                crop_stage_data.append({
                    'plot_no': plot_no,
                    'Latitude':Latitude,
                    'Longitude':Longitude,
                    'planting_date':planting_date,
                    'Crop_Stage': stage,
                    'Stage_Date': stage_date_value,
                    'GDD': stage_gdd_value,
                    'Ideal_GDD': gdd_threshold  # Add the Ideal GDD value here
                })

    # Convert the list to a DataFrame
    crop_stage_df = pd.DataFrame(crop_stage_data)

    return crop_stage_df

```


```python
vcross_df=determine_corn_crop_stages_with_ideal_gdd(merged_df)
vcross_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plot_no</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>planting_date</th>
      <th>Crop_Stage</th>
      <th>Stage_Date</th>
      <th>GDD</th>
      <th>Ideal_GDD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-07</td>
      <td>Emergence (VE)</td>
      <td>2021-10-16</td>
      <td>112.30</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-07</td>
      <td>V2 (Two leaves with collars visible)</td>
      <td>2021-10-29</td>
      <td>259.90</td>
      <td>250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-07</td>
      <td>V6 (Six leaves with collars visible)</td>
      <td>2021-11-17</td>
      <td>477.50</td>
      <td>475</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-07</td>
      <td>VT (Tasseling)</td>
      <td>2021-12-29</td>
      <td>903.40</td>
      <td>900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>-0.499127</td>
      <td>37.612253</td>
      <td>2021-10-07</td>
      <td>R1 (Silking)</td>
      <td>2022-01-18</td>
      <td>1101.65</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2364</th>
      <td>428</td>
      <td>-0.774969</td>
      <td>37.066322</td>
      <td>2021-10-22</td>
      <td>Emergence (VE)</td>
      <td>2021-10-30</td>
      <td>100.20</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2365</th>
      <td>428</td>
      <td>-0.774969</td>
      <td>37.066322</td>
      <td>2021-10-22</td>
      <td>V2 (Two leaves with collars visible)</td>
      <td>2021-11-13</td>
      <td>254.70</td>
      <td>250</td>
    </tr>
    <tr>
      <th>2366</th>
      <td>428</td>
      <td>-0.774969</td>
      <td>37.066322</td>
      <td>2021-10-22</td>
      <td>V6 (Six leaves with collars visible)</td>
      <td>2021-12-05</td>
      <td>484.70</td>
      <td>475</td>
    </tr>
    <tr>
      <th>2367</th>
      <td>428</td>
      <td>-0.774969</td>
      <td>37.066322</td>
      <td>2021-10-22</td>
      <td>VT (Tasseling)</td>
      <td>2022-01-17</td>
      <td>901.45</td>
      <td>900</td>
    </tr>
    <tr>
      <th>2368</th>
      <td>428</td>
      <td>-0.774969</td>
      <td>37.066322</td>
      <td>2021-10-22</td>
      <td>R1 (Silking)</td>
      <td>2022-02-06</td>
      <td>1110.30</td>
      <td>1100</td>
    </tr>
  </tbody>
</table>
<p>2369 rows × 8 columns</p>
</div>




```python
stage_output="crop_stages_kenya.csv"
vcross_df.to_csv(stage_output,index=False)
```


```python

```


```python
import pandas as pd

# --- Assume you already have these two DataFrames ---
# weather_df  (with Data_Planting_Date and Planting_Date)
# stage_df    (with Crop_Stage and Stage_Date)

# 1️⃣ Normalize column names
merged_df.rename(columns={'Plot_No': 'plot_no'}, inplace=True)
vcross_df['Stage_Date'] = pd.to_datetime(vcross_df['Stage_Date'])
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df['Planting_Date'] = pd.to_datetime(merged_df['Planting_Date'])

# 2️⃣ Get latest stage per plot
latest_stage = vcross_df.sort_values('Stage_Date').groupby('plot_no').tail(1)
latest_stage = latest_stage[['plot_no', 'Crop_Stage', 'Stage_Date']]

# 3️⃣ Prepare an empty list to hold summary records
records = []

# 4️⃣ Loop through each plot and compute aggregated weather metrics
for _, row in latest_stage.iterrows():
    plot = row['plot_no']
    latest_stage_date = row['Stage_Date']
    crop_stage = row['Crop_Stage']

    # planting date for that plot
    planting_date = merged_df.loc[merged_df['plot_no'] == plot, 'Planting_Date'].iloc[0]

    # filter weather data within that range
    w = merged_df[(merged_df['plot_no'] == plot) &
                   (merged_df['Date'] >= planting_date) &
                   (merged_df['Date'] <= latest_stage_date)]

    if len(w) == 0:
        continue  # skip if no data

    # Aggregate
    record = {
        'plot_no': plot,
        'latitude': w['Latitude'].iloc[0],
        'longitude': w['Longitude'].iloc[0],
        'Data_Planting_Date': w['Data_Planting_Date'].iloc[0],
        'Planting_Date': planting_date,
        'Latest_Stage': crop_stage,
        'Latest_Stage_Date': latest_stage_date,
        'Max_Temp_C': w['Max_Temp_C'].max(),
        'Min_Temp_C': w['Min_Temp_C'].min(),
        'Mean_Temp_C': w['Mean_Temp_C'].mean(),
        'Precip_mm': w['Precip_mm'].sum(),
        'Humidity_%': w['Humidity_%'].mean(),
        'SolarEnergy_MJm2': w['SolarEnergy_MJm2'].sum(),
        'Evapotranspiration_mm': w['Evapotranspiration_mm'].sum(skipna=True),
        'WindSpeed_kmh': w['WindSpeed_kmh'].mean(),
        'CloudCover_%': w['CloudCover_%'].mean()
    }

    records.append(record)

# 5️⃣ Create final summary DataFrame
summary_df = pd.DataFrame(records)

# 6️⃣ Preview
summary_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plot_no</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Data_Planting_Date</th>
      <th>Planting_Date</th>
      <th>Latest_Stage</th>
      <th>Latest_Stage_Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>Evapotranspiration_mm</th>
      <th>WindSpeed_kmh</th>
      <th>CloudCover_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145</td>
      <td>0.118367</td>
      <td>37.728812</td>
      <td>10/7/2020</td>
      <td>2020-09-05</td>
      <td>R6 (Maturity)</td>
      <td>2021-05-22</td>
      <td>30.8</td>
      <td>8.9</td>
      <td>20.950439</td>
      <td>1329.017</td>
      <td>72.812719</td>
      <td>4971.9</td>
      <td>0.0</td>
      <td>12.819298</td>
      <td>52.488158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139</td>
      <td>0.219597</td>
      <td>37.788483</td>
      <td>10/10/2020</td>
      <td>2020-09-06</td>
      <td>R6 (Maturity)</td>
      <td>2021-05-23</td>
      <td>30.7</td>
      <td>8.9</td>
      <td>20.867257</td>
      <td>1323.000</td>
      <td>72.303982</td>
      <td>4966.7</td>
      <td>0.0</td>
      <td>13.486726</td>
      <td>52.800885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>230</td>
      <td>0.302884</td>
      <td>38.085174</td>
      <td>10/20/2020</td>
      <td>2020-09-04</td>
      <td>R6 (Maturity)</td>
      <td>2021-05-23</td>
      <td>32.2</td>
      <td>8.9</td>
      <td>21.790278</td>
      <td>1319.300</td>
      <td>74.532870</td>
      <td>4726.7</td>
      <td>0.0</td>
      <td>13.418519</td>
      <td>54.375000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>411</td>
      <td>-0.897704</td>
      <td>37.026625</td>
      <td>10/14/2020</td>
      <td>2020-09-08</td>
      <td>R6 (Maturity)</td>
      <td>2021-06-04</td>
      <td>29.7</td>
      <td>9.7</td>
      <td>20.076068</td>
      <td>829.114</td>
      <td>71.237607</td>
      <td>4887.9</td>
      <td>0.0</td>
      <td>24.261538</td>
      <td>67.326923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>-0.473318</td>
      <td>37.415947</td>
      <td>10/18/2020</td>
      <td>2020-10-14</td>
      <td>R6 (Maturity)</td>
      <td>2021-06-07</td>
      <td>30.1</td>
      <td>9.6</td>
      <td>20.512446</td>
      <td>1235.535</td>
      <td>72.881116</td>
      <td>4694.1</td>
      <td>0.0</td>
      <td>12.799142</td>
      <td>60.377682</td>
    </tr>
  </tbody>
</table>
</div>




```python
unique_plots = summary_df['plot_no'].nunique()
unique_plots
```




    412




```python
# --- Merge harvest data into summary_df ---
final_df = pd.merge(
    summary_df,
    df[['plot_no', 'crop_type', 'wet_harvest_kg/ha', 'dry_harvest_kg/ha']],
    on=["plot_no"],
    how='left'
)

# --- Preview final dataframe ---
final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plot_no</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Data_Planting_Date</th>
      <th>Planting_Date</th>
      <th>Latest_Stage</th>
      <th>Latest_Stage_Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>Evapotranspiration_mm</th>
      <th>WindSpeed_kmh</th>
      <th>CloudCover_%</th>
      <th>crop_type</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145</td>
      <td>0.118367</td>
      <td>37.728812</td>
      <td>10/7/2020</td>
      <td>2020-09-05</td>
      <td>R6 (Maturity)</td>
      <td>2021-05-22</td>
      <td>30.8</td>
      <td>8.9</td>
      <td>20.950439</td>
      <td>1329.017</td>
      <td>72.812719</td>
      <td>4971.9</td>
      <td>0.0</td>
      <td>12.819298</td>
      <td>52.488158</td>
      <td>maize</td>
      <td>2853.75</td>
      <td>2328.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139</td>
      <td>0.219597</td>
      <td>37.788483</td>
      <td>10/10/2020</td>
      <td>2020-09-06</td>
      <td>R6 (Maturity)</td>
      <td>2021-05-23</td>
      <td>30.7</td>
      <td>8.9</td>
      <td>20.867257</td>
      <td>1323.000</td>
      <td>72.303982</td>
      <td>4966.7</td>
      <td>0.0</td>
      <td>13.486726</td>
      <td>52.800885</td>
      <td>maize</td>
      <td>526.88</td>
      <td>198.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>230</td>
      <td>0.302884</td>
      <td>38.085174</td>
      <td>10/20/2020</td>
      <td>2020-09-04</td>
      <td>R6 (Maturity)</td>
      <td>2021-05-23</td>
      <td>32.2</td>
      <td>8.9</td>
      <td>21.790278</td>
      <td>1319.300</td>
      <td>74.532870</td>
      <td>4726.7</td>
      <td>0.0</td>
      <td>13.418519</td>
      <td>54.375000</td>
      <td>maize</td>
      <td>1402.50</td>
      <td>1006.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>411</td>
      <td>-0.897704</td>
      <td>37.026625</td>
      <td>10/14/2020</td>
      <td>2020-09-08</td>
      <td>R6 (Maturity)</td>
      <td>2021-06-04</td>
      <td>29.7</td>
      <td>9.7</td>
      <td>20.076068</td>
      <td>829.114</td>
      <td>71.237607</td>
      <td>4887.9</td>
      <td>0.0</td>
      <td>24.261538</td>
      <td>67.326923</td>
      <td>maize</td>
      <td>5417.50</td>
      <td>2305.62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>-0.473318</td>
      <td>37.415947</td>
      <td>10/18/2020</td>
      <td>2020-10-14</td>
      <td>R6 (Maturity)</td>
      <td>2021-06-07</td>
      <td>30.1</td>
      <td>9.6</td>
      <td>20.512446</td>
      <td>1235.535</td>
      <td>72.881116</td>
      <td>4694.1</td>
      <td>0.0</td>
      <td>12.799142</td>
      <td>60.377682</td>
      <td>maize</td>
      <td>4610.00</td>
      <td>2598.12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the unwanted columns
final_for_yield = final_df.drop(columns=[
    "Data_Planting_Date",
    "Latest_Stage",
    "Latest_Stage_Date",
    "Evapotranspiration_mm"
])
```


```python
final_for_yield.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plot_no</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Planting_Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>WindSpeed_kmh</th>
      <th>CloudCover_%</th>
      <th>crop_type</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145</td>
      <td>0.118367</td>
      <td>37.728812</td>
      <td>2020-09-05</td>
      <td>30.8</td>
      <td>8.9</td>
      <td>20.950439</td>
      <td>1329.017</td>
      <td>72.812719</td>
      <td>4971.9</td>
      <td>12.819298</td>
      <td>52.488158</td>
      <td>maize</td>
      <td>2853.75</td>
      <td>2328.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139</td>
      <td>0.219597</td>
      <td>37.788483</td>
      <td>2020-09-06</td>
      <td>30.7</td>
      <td>8.9</td>
      <td>20.867257</td>
      <td>1323.000</td>
      <td>72.303982</td>
      <td>4966.7</td>
      <td>13.486726</td>
      <td>52.800885</td>
      <td>maize</td>
      <td>526.88</td>
      <td>198.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>230</td>
      <td>0.302884</td>
      <td>38.085174</td>
      <td>2020-09-04</td>
      <td>32.2</td>
      <td>8.9</td>
      <td>21.790278</td>
      <td>1319.300</td>
      <td>74.532870</td>
      <td>4726.7</td>
      <td>13.418519</td>
      <td>54.375000</td>
      <td>maize</td>
      <td>1402.50</td>
      <td>1006.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>411</td>
      <td>-0.897704</td>
      <td>37.026625</td>
      <td>2020-09-08</td>
      <td>29.7</td>
      <td>9.7</td>
      <td>20.076068</td>
      <td>829.114</td>
      <td>71.237607</td>
      <td>4887.9</td>
      <td>24.261538</td>
      <td>67.326923</td>
      <td>maize</td>
      <td>5417.50</td>
      <td>2305.62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>-0.473318</td>
      <td>37.415947</td>
      <td>2020-10-14</td>
      <td>30.1</td>
      <td>9.6</td>
      <td>20.512446</td>
      <td>1235.535</td>
      <td>72.881116</td>
      <td>4694.1</td>
      <td>12.799142</td>
      <td>60.377682</td>
      <td>maize</td>
      <td>4610.00</td>
      <td>2598.12</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_for_yield.to_csv('final_for_yield.csv')
```

**Import Indices**


```python
index_df=pd.read_csv('all_indices.csv')
index_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Date</th>
      <th>Mean_NDVI</th>
      <th>Mean_EVI</th>
      <th>Mean_SAVI</th>
      <th>Mean_NDRE</th>
      <th>Mean_NDWI</th>
      <th>Mean_NDMI</th>
      <th>Max_NDVI</th>
      <th>Max_EVI</th>
      <th>...</th>
      <th>Min_NDRE</th>
      <th>Min_NDWI</th>
      <th>Min_NDMI</th>
      <th>Std_Dev_NDVI</th>
      <th>Std_Dev_EVI</th>
      <th>Std_Dev_SAVI</th>
      <th>Std_Dev_NDRE</th>
      <th>Std_Dev_NDWI</th>
      <th>Std_Dev_NDMI</th>
      <th>Cumulative_NDVI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>10/07/2021</td>
      <td>0.315456</td>
      <td>0.169707</td>
      <td>0.473113</td>
      <td>0.193602</td>
      <td>-0.495265</td>
      <td>-0.074549</td>
      <td>0.332959</td>
      <td>0.175669</td>
      <td>...</td>
      <td>0.167763</td>
      <td>-0.507765</td>
      <td>-0.095315</td>
      <td>0.026222</td>
      <td>0.015488</td>
      <td>0.039328</td>
      <td>0.015306</td>
      <td>0.014207</td>
      <td>0.009093</td>
      <td>0.3155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>10/12/2021</td>
      <td>0.315456</td>
      <td>0.169707</td>
      <td>0.473113</td>
      <td>0.193602</td>
      <td>-0.495265</td>
      <td>-0.074549</td>
      <td>0.332959</td>
      <td>0.175669</td>
      <td>...</td>
      <td>0.167763</td>
      <td>-0.507765</td>
      <td>-0.095315</td>
      <td>0.026222</td>
      <td>0.015488</td>
      <td>0.039328</td>
      <td>0.015306</td>
      <td>0.014207</td>
      <td>0.009093</td>
      <td>0.6309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10/17/2021</td>
      <td>0.333320</td>
      <td>0.171813</td>
      <td>0.499899</td>
      <td>0.202536</td>
      <td>-0.511514</td>
      <td>-0.082881</td>
      <td>0.346773</td>
      <td>0.174626</td>
      <td>...</td>
      <td>0.178060</td>
      <td>-0.523987</td>
      <td>-0.101538</td>
      <td>0.024542</td>
      <td>0.013931</td>
      <td>0.036807</td>
      <td>0.012043</td>
      <td>0.013925</td>
      <td>0.008150</td>
      <td>0.9642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>10/22/2021</td>
      <td>0.340673</td>
      <td>0.184697</td>
      <td>0.510929</td>
      <td>0.208338</td>
      <td>-0.504088</td>
      <td>-0.078778</td>
      <td>0.354052</td>
      <td>0.187796</td>
      <td>...</td>
      <td>0.186298</td>
      <td>-0.516319</td>
      <td>-0.096873</td>
      <td>0.023275</td>
      <td>0.013660</td>
      <td>0.034907</td>
      <td>0.012426</td>
      <td>0.013476</td>
      <td>0.009825</td>
      <td>1.3049</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>10/27/2021</td>
      <td>0.352110</td>
      <td>0.188441</td>
      <td>0.528080</td>
      <td>0.215876</td>
      <td>-0.507386</td>
      <td>-0.077881</td>
      <td>0.360551</td>
      <td>0.188966</td>
      <td>...</td>
      <td>0.194217</td>
      <td>-0.517190</td>
      <td>-0.095863</td>
      <td>0.019946</td>
      <td>0.011861</td>
      <td>0.029915</td>
      <td>0.012517</td>
      <td>0.011916</td>
      <td>0.010360</td>
      <td>1.6570</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
df = index_df.copy()
df['Date'] = pd.to_datetime(df['Date'])
```


```python
df = df.sort_values(['Plot_No', 'Date'])
# Columns that start with "Mean_"
mean_cols = [col for col in df.columns if col.startswith("Mean_")]
```


```python
agg_dict = {col: 'sum' for col in mean_cols}
agg_dict['Cumulative_NDVI'] = 'last'
```


```python
# Group by Plot_No
final_df = df.groupby('Plot_No').agg(agg_dict).reset_index()

final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Plot_No</th>
      <th>Mean_NDVI</th>
      <th>Mean_EVI</th>
      <th>Mean_SAVI</th>
      <th>Mean_NDRE</th>
      <th>Mean_NDWI</th>
      <th>Mean_NDMI</th>
      <th>Cumulative_NDVI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6.819444</td>
      <td>3.719848</td>
      <td>10.227529</td>
      <td>4.463286</td>
      <td>-7.211514</td>
      <td>1.572886</td>
      <td>6.8194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>9.580029</td>
      <td>5.905028</td>
      <td>14.366404</td>
      <td>6.331487</td>
      <td>-8.904260</td>
      <td>3.485063</td>
      <td>9.5800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.781868</td>
      <td>2.845189</td>
      <td>7.171665</td>
      <td>2.973770</td>
      <td>-6.208784</td>
      <td>-0.301703</td>
      <td>4.7819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8.503830</td>
      <td>5.336094</td>
      <td>12.753115</td>
      <td>5.995714</td>
      <td>-8.596676</td>
      <td>0.985690</td>
      <td>8.5038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9.004005</td>
      <td>5.087394</td>
      <td>13.502884</td>
      <td>6.120798</td>
      <td>-8.974788</td>
      <td>2.486243</td>
      <td>9.0040</td>
    </tr>
  </tbody>
</table>
</div>




```python
agg_df = final_df.rename(columns={'Plot_No': 'plot_no'})
```

# **Merge the index and weathe df**


```python
merged_df = final_for_yield.merge(agg_df, on='plot_no', how='left')
merged_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plot_no</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Planting_Date</th>
      <th>Max_Temp_C</th>
      <th>Min_Temp_C</th>
      <th>Mean_Temp_C</th>
      <th>Precip_mm</th>
      <th>Humidity_%</th>
      <th>SolarEnergy_MJm2</th>
      <th>...</th>
      <th>crop_type</th>
      <th>wet_harvest_kg/ha</th>
      <th>dry_harvest_kg/ha</th>
      <th>Mean_NDVI</th>
      <th>Mean_EVI</th>
      <th>Mean_SAVI</th>
      <th>Mean_NDRE</th>
      <th>Mean_NDWI</th>
      <th>Mean_NDMI</th>
      <th>Cumulative_NDVI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145</td>
      <td>0.118367</td>
      <td>37.728812</td>
      <td>2020-09-05</td>
      <td>30.8</td>
      <td>8.9</td>
      <td>20.950439</td>
      <td>1329.017</td>
      <td>72.812719</td>
      <td>4971.9</td>
      <td>...</td>
      <td>maize</td>
      <td>2853.75</td>
      <td>2328.75</td>
      <td>8.050688</td>
      <td>11.513297</td>
      <td>12.073975</td>
      <td>5.325104</td>
      <td>-8.078459</td>
      <td>2.951477</td>
      <td>8.0507</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139</td>
      <td>0.219597</td>
      <td>37.788483</td>
      <td>2020-09-06</td>
      <td>30.7</td>
      <td>8.9</td>
      <td>20.867257</td>
      <td>1323.000</td>
      <td>72.303982</td>
      <td>4966.7</td>
      <td>...</td>
      <td>maize</td>
      <td>526.88</td>
      <td>198.75</td>
      <td>6.343849</td>
      <td>3.194971</td>
      <td>9.513950</td>
      <td>3.798797</td>
      <td>-7.624553</td>
      <td>2.439587</td>
      <td>6.3438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>230</td>
      <td>0.302884</td>
      <td>38.085174</td>
      <td>2020-09-04</td>
      <td>32.2</td>
      <td>8.9</td>
      <td>21.790278</td>
      <td>1319.300</td>
      <td>74.532870</td>
      <td>4726.7</td>
      <td>...</td>
      <td>maize</td>
      <td>1402.50</td>
      <td>1006.25</td>
      <td>11.314518</td>
      <td>6.294860</td>
      <td>16.968715</td>
      <td>7.431315</td>
      <td>-11.223136</td>
      <td>3.638672</td>
      <td>11.3145</td>
    </tr>
    <tr>
      <th>3</th>
      <td>411</td>
      <td>-0.897704</td>
      <td>37.026625</td>
      <td>2020-09-08</td>
      <td>29.7</td>
      <td>9.7</td>
      <td>20.076068</td>
      <td>829.114</td>
      <td>71.237607</td>
      <td>4887.9</td>
      <td>...</td>
      <td>maize</td>
      <td>5417.50</td>
      <td>2305.62</td>
      <td>11.743474</td>
      <td>8.983573</td>
      <td>17.612568</td>
      <td>7.812345</td>
      <td>-10.966121</td>
      <td>5.535510</td>
      <td>11.7435</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>-0.473318</td>
      <td>37.415947</td>
      <td>2020-10-14</td>
      <td>30.1</td>
      <td>9.6</td>
      <td>20.512446</td>
      <td>1235.535</td>
      <td>72.881116</td>
      <td>4694.1</td>
      <td>...</td>
      <td>maize</td>
      <td>4610.00</td>
      <td>2598.12</td>
      <td>7.560659</td>
      <td>8.125853</td>
      <td>11.339539</td>
      <td>5.216997</td>
      <td>-6.901054</td>
      <td>3.690113</td>
      <td>7.5607</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
display(merged_df.columns)

```


    Index(['plot_no', 'latitude', 'longitude', 'Planting_Date', 'Max_Temp_C',
           'Min_Temp_C', 'Mean_Temp_C', 'Precip_mm', 'Humidity_%',
           'SolarEnergy_MJm2', 'WindSpeed_kmh', 'CloudCover_%', 'crop_type',
           'wet_harvest_kg/ha', 'dry_harvest_kg/ha', 'Mean_NDVI', 'Mean_EVI',
           'Mean_SAVI', 'Mean_NDRE', 'Mean_NDWI', 'Mean_NDMI', 'Cumulative_NDVI'],
          dtype='object')



```python
merged_df.to_csv('final_dataset.csv')
```


```python
import nbformat
from nbconvert import MarkdownExporter

with open('yield_features.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

exporter = MarkdownExporter()
(body, _) = exporter.from_notebook_node(nb)
body[:2000]

```


    ---------------------------------------------------------------------------

    UnicodeDecodeError                        Traceback (most recent call last)

    Cell In[2], line 5
          2 from nbconvert import MarkdownExporter
          4 with open('yield_features.ipynb') as f:
    ----> 5     nb = nbformat.read(f, as_version=4)
          7 exporter = MarkdownExporter()
          8 (body, _) = exporter.from_notebook_node(nb)
    

    File ~\anaconda3\Lib\site-packages\nbformat\__init__.py:169, in read(fp, as_version, capture_validation_error, **kwargs)
        141 """Read a notebook from a file as a NotebookNode of the given version.
        142 
        143 The string can contain a notebook of any version.
       (...)
        165     The notebook that was read.
        166 """
        168 try:
    --> 169     buf = fp.read()
        170 except AttributeError:
        171     with open(fp, encoding="utf8") as f:  # noqa: PTH123
    

    File ~\anaconda3\Lib\encodings\cp1252.py:23, in IncrementalDecoder.decode(self, input, final)
         22 def decode(self, input, final=False):
    ---> 23     return codecs.charmap_decode(input,self.errors,decoding_table)[0]
    

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 69818: character maps to <undefined>



```python

```
