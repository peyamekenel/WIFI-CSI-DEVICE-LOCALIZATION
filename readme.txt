Data statement:
The following dataset was collected as part of the OPERA Project, the UK Engineering and Physical Sciences Research Council (EPSRC), Grant EP/R018677/1.

*********************************************************************************************************************************************************
Description of WiFi Channel State Information (CSI) device-to-device localisation experiment:

Dataset was collected using two WiFi devices: 
(i)  one transmitter (Next Unit of Computing (NUC) device equipped with an Intel 5300 network interface card (NIC)) 
(ii) one receiver    (Next Unit of Computing (NUC) device equipped with an Intel 5300 network interface card (NIC))

Complex CSI data was recorded at the receiver using the monitor mode.
see https://dhalperi.github.io/linux-80211n-csitool/ for more information on CSI and installation instructions.


Parameters:
(1) No participants - Dataset intended for device-to-device localisation using CSI data.
(2) Line-of-sight between transmitter and receiver.
(3) Transmitter transmits data with one antenna, receiver captures data on 3 antennas.
(4) 5 GHz WiFi band, channel 64, 5320 MHz, 40 MHz channel bandwidth, 30 OFDM subcarriers.
(5) Packet rate = 2500 Hz.
(6) Transmitter was placed at 2 different angles with respect to the receiver: 30 degrees and -60 degrees and at 5 different distances: 1m, 2m, 3m, 4m, 5m
(see "Setup_and_Parameters.pdf" for more details)


The zipped dataset folder "wifi_csi_data_loc.zip" consists of 10 .mat (MATLAB) files:

(1) "loc_30deg_1m.mat"		: Transmitter at 30 degrees with respect to receiver and separation between the two devices is 1 metre.
(2) "loc_30deg_2m.mat"		: Transmitter at 30 degrees with respect to receiver and separation between the two devices is 2 metres.
(3) "loc_30deg_3m.mat"		: Transmitter at 30 degrees with respect to receiver and separation between the two devices is 3 metres.
(4) "loc_30deg_4m.mat"		: Transmitter at 30 degrees with respect to receiver and separation between the two devices is 4 metres.
(5) "loc_30deg_5m.mat"		: Transmitter at 30 degrees with respect to receiver and separation between the two devices is 5 metres.
(6) "loc_minus60deg_1m.mat"	: Transmitter at -60 degrees with respect to receiver and separation between the two devices is 1 metre.
(7) "loc_minus60deg_2m.mat"	: Transmitter at -60 degrees with respect to receiver and separation between the two devices is 2 metres.
(8) "loc_minus60deg_3m.mat"	: Transmitter at -60 degrees with respect to receiver and separation between the two devices is 3 metres.
(9) "loc_minus60deg_4m.mat"	: Transmitter at -60 degrees with respect to receiver and separation between the two devices is 4 metres.
(10)"loc_minus60deg_5m.mat"	: Transmitter at -60 degrees with respect to receiver and separation between the two devices is 5 metres.

The data contained within each .mat file is described below:

Within each .mat file, the raw complex CSI data is organised as a 3D matrix: 3 x 30 x number of packets (see naming "csi_complex_data" when loading each dataset file in MATLAB).
This corresponds to 3 receiving antennas x 30 OFDM subcarriers x number of packets.
The Intel 5300 NIC extracts CSI data over only 30 subcarriers.
The CSI data has NOT been processed in any way (raw data).

Software required:
MATLAB software is required to open the dataset files.
