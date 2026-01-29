# Import dependencies
from obspy import UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth
from toolbox import read_sds, add_station_metadata, check_timeline

# Define variables for SDS data reading
SDS_ROOT = "/VDAP-NAS/jwellik/DATA/CVGHM/IBU/SDS/"  # Update with your SDS root directory
NETWORK = "VG"
STATION = "GIN,IBKC,IBTB,IBTL"  # Update with your station codes
LOCATION = "00,1L,1L,1L"
CHANNEL = "EHZ,HHZ,HHZ,HHZ"

STARTTIME = UTCDateTime(2025, 1, 1, 0, 0)  # Update with your start time
ENDTIME = STARTTIME + 3*3600  # 3 hours
PAD = 360  # s

# Coordinate file path (JSON format: {"STATION_CODE": [latitude, longitude, elevation], ...})
COORD_FILEPATH = "./ibu_station.json"

# Define variables for check_timeline function
OVERLAP = 0.5  # 1 min time step for 1 min interval
GENERATE_FIG = True
FIG_WIDTH = 8
FIG_HEIGHT = 6
FONT_S = 8
MODEL_PATH = "./models/voissnet_seismic_generalized_model.keras"
MEANVAR_PATH = ""
PNORM_THRESH = 0.4  # threshold for p-norm, can be None
SPEC_KWARGS = None
EXPORT_PATH = "./output/ibu/"
TRANSPARENT = None

# Volcano coordinates (update with your volcano location)
VOLC_COORDS = (1.4941, 127.6364)  # Update with (latitude, longitude)

# [Frequency Index] Set up FI kwargs (DR requires response removal, so use FI instead)
FI_KWARGS = {"reference_station": "all",        # station code or "all"
             "window_length": 10,               # seconds
             "overlap": 0.5,                    # fraction of window length
             "filomin": 1,                      # Hz -- FI lower band minimum
             "filomax": 2.5,                    # Hz -- FI lower band maximum
             "fiupmin": 5,                      # Hz -- FI upper band minimum
             "fiupmax": 10,                     # Hz -- FI upper band maximum
             "med_filt_kernel": None,           # Kernel size for median filter smoothing
             "volc_lat": VOLC_COORDS[0],        # decimal degrees (only for source-station distance in y-label)
             "volc_lon": VOLC_COORDS[1]}        # decimal degrees (only for source-station distance in y-label)

# Read data from SDS archive
stream = read_sds(SDS_ROOT,
                  NETWORK,
                  STATION,
                  LOCATION,
                  CHANNEL,
                  STARTTIME - PAD,
                  ENDTIME + PAD,
                  merge=-1,
                  verbose=True)

# Add station coordinates from JSON file
stream = add_station_metadata(stream, COORD_FILEPATH, verbose=True)

# Sort stream by distance to volcano (order of the stream input determines order of subplots)
if len(stream) > 0 and hasattr(stream[0].stats, 'latitude'):
    stream.traces.sort(key=lambda tr: gps2dist_azimuth(VOLC_COORDS[0], VOLC_COORDS[1], 
                                                       tr.stats.latitude, tr.stats.longitude)[0])

# Run VOISS-Net WITHOUT response removal
class_mat, prob_mat = check_timeline(stream,
                                     STARTTIME,
                                     ENDTIME,
                                     MODEL_PATH,
                                     MEANVAR_PATH,
                                     OVERLAP,
                                     pnorm_thresh=PNORM_THRESH,
                                     generate_fig=GENERATE_FIG,
                                     fig_width=FIG_WIDTH,
                                     fig_height=FIG_HEIGHT,
                                     font_s=FONT_S,
                                     spec_kwargs=SPEC_KWARGS,
                                     fi_kwargs=FI_KWARGS,  # Use FI instead of DR (DR requires response removal)
                                     export_path=EXPORT_PATH,
                                     transparent=TRANSPARENT,
                                     remove_response=False)  # Set to False since we don't have response info
