
import gmaps
import gmaps.datasets
gmaps.configure(api_key='AI...')

df = gmaps.datasets.load_dataset_as_df('earthquakes')
# dataframe with columns ('latitude', 'longitude', 'magnitude')

fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(
    df[['latitude', 'longitude']], weights=df['magnitude'],
    max_intensity=30, point_radius=3.0
)
fig.add_layer(heatmap_layer)
fig
