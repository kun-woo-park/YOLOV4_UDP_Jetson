import torch
import socket
import numpy as np
import torch.nn as nn
from torch import Tensor

# model load
tr_model = torch.load("./Custom_model_fin")
tr_model.eval()
mean = np.load('mean.npy')
std = np.load('std.npy')

#######################################################################################
################# Unity packet data information (from ASCL, Inha Univ.) ###############

# setting constants
Deg2Rad = np.pi / 180
g = 9.8
K_alt = 1.0
Deg2Rad = np.pi / 180
dist_sep = 100

# initialize dynamics variables
ac = 0
r = 1000
vc = 100
elev = 0
azim = 0
hdot_cmd = 0
los = 0
los_p = los
los_p2 = los_p
dlos = 0
dlos_p = dlos
dlos_p2 = dlos_p
azim = 0
azim_p = azim
azim_p2 = azim_p
daz = 0
daz_p = daz
daz_p2 = daz_p

# received packet number
packetCounter = 0

# aircraft & sensor models update state counters, increases by 1 each time the model updates it's state
RADALT_update_couter = 0
ATAR_update_couter = 0
ATAR_update_couter_p = 0
ADSB_update_counter = 0
OWNship_update_counter = 0
Intruder_update_counter = 0
Frame_update_counter = 0

# radio altimeter data (sensor) _ OWN-ship altitude
RADALT_vertical_rate = 0
RADALT_ground_altitude = 0

# Air_to_Air_Radar data (sensor)
ATAR_intruder_bearing = 0
ATAR_intruder_elevation = 0
ATAR_intruder_closing_velocity = 0
ATAR_intruder_relative_distance = 100000
ATAR_intruder_detection_flag = False

# ADSB data (sensor)
ADSB_intruder_longitude = 0
ADSB_intruder_latitude = 0
ADSB_intruder_altitude = 0
ADSB_intruder_heading = 0
ADSB_intruder_velocity = 0
ADSB_intruder_vertical_rate = 0
ADSB_intruder_callsign = " "
ADSB_intruder_iCAOadd = " "

# OWN-ship states (dynamic model)
OWNship_longitude = 0
Ownship_latitude = 0
OWNship_altitude = 0
OWNship_heading = 0
OWNship_velocity = 0
OWNship_pos_x = 0
OWNship_pos_y = 0
OWNship_pos_z = 0
OWNship_roll = 0
OWNship_pitch = 0
OWNship_yaw = 0
OWNship_vertical_rate = 0

# INTruder states (dynamic model)
INTruder_longitude = 0
INTruderlatitude = 0
INTruderaltitude = 0
INTruder_heading = 0
INTruder_velocity = 0
INTruder_pos_x = 0
INTruder_pos_y = 0
INTruder_pos_z = 0
INTruder_roll = 0
INTruder_pitch = 0
INTruder_yaw = 0
INTruder_vertical_rate = 0

#############################################

######## packet ICD (python to unity) #######

header1 = 80
header2 = 85
header3 = 67

flightMode = 1  # mode 1 = auto , mode 0 = manual

rollCMD = 0  # 0~100~200 (roll rate)
alphaCMD = 0  # 0~100~200 (pitch rate)
thrustCMD = 0  # 0~100~200 (0 being 0 thrust and 200 full thrust )

PanCMD = 0
TiltCMD = 0
lockModeCMD = 0
ZominCMD = 0
ZomoutCMD = 0
cameraLselect = 1
cameraRselect = 0

cmdPacket = bytearray(7)
cameraCmdPacket = bytearray(10)


#######################################################################################
#######################################################################################


class FClayer(nn.Module):  # define fully connected layer with Leaky ReLU activation function
    def __init__(self, innodes, nodes):
        super(FClayer, self).__init__()
        self.fc = nn.Linear(innodes, nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.act(out)
        return out


class WaveNET(nn.Module):  # define custom model named wave net, which was coined after seeing the nodes sway
    def __init__(self, block, planes, nodes, num_classes=3):
        super(WaveNET, self).__init__()
        self.innodes = 5

        self.layer1 = self._make_layer(block, planes[0], nodes[0])
        self.layer2 = self._make_layer(block, planes[1], nodes[1])
        self.layer3 = self._make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.innodes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, nodes):

        layers = []
        layers.append(block(self.innodes, nodes))
        self.innodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.innodes, nodes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# sending avoidance command to Unity simulator
def setCMD(client_ip, client_port, mode, roll, alpha, thrust, DBPAStakeOver):
    cmdPacket[0] = header1
    cmdPacket[1] = header2
    cmdPacket[2] = mode
    cmdPacket[3] = roll
    alpha = int(alpha)
    cmdPacket[4] = alpha
    cmdPacket[5] = thrust
    cmdPacket[6] = DBPAStakeOver
    sock.sendto(cmdPacket, (client_ip, client_port))           # send command packet to IP for Unity simulator


def UDP_main(host_ip, host_port, client_ip, client_port, yolo_port):
    results = []                                                    # list for print
    detection_flag = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host_ip, host_port))
    sock.settimeout(2)

    while True:
        if detection_flag == 0:
            try:
                received_data, addr = sock.recvfrom(1024)

                sock_local = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_local.bind(("127.0.0.1", yolo_port))
                flag, addr = sock_local.recvfrom(1024)
                detection_flag = 1
                print(flag.decode())
                sock_local.close()
            except socket.timeout:
                print("timeout 0")
        else:
            try:
                received_data, addr = sock.recvfrom(1024)
                SILS_stat = []
                deliminatorindex = -1

                for char in range(0, len(received_data)):
                    if received_data[char] == 32:
                        SILS_stat.append(received_data[deliminatorindex + 1:char])
                        deliminatorindex = char

                RADALT_update_couter = float(SILS_stat[1])
                ATAR_update_couter = float(SILS_stat[2])
                ADSB_update_counter = float(SILS_stat[3])
                OWNship_update_counter = float(SILS_stat[4])
                Intruder_update_counter = float(SILS_stat[5])
                Frame_update_counter = float(SILS_stat[6])

                RADALT_vertical_rate = float(SILS_stat[7])
                RADALT_ground_altitude = float(SILS_stat[8])

                ATAR_intruder_bearing = float(SILS_stat[9])
                ATAR_intruder_elevation = float(SILS_stat[10])
                ATAR_intruder_relative_distance = float(SILS_stat[11])
                ATAR_intruder_closing_velocity = float(SILS_stat[11 + 1])
                ATAR_intruder_detection_flag = SILS_stat[12 + 1]

                ADSB_intruder_longitude = float(SILS_stat[13 + 1])
                ADSB_intruder_latitude = float(SILS_stat[14 + 1])
                ADSB_intruder_altitude = float(SILS_stat[15 + 1])
                ADSB_intruder_heading = float(SILS_stat[16 + 1])
                ADSB_intruder_velocity = float(SILS_stat[17 + 1])
                ADSB_intruder_vertical_rate = float(SILS_stat[18 + 1])
                ADSB_intruder_callsign = SILS_stat[19 + 1]
                ADSB_intruder_iCAOadd = SILS_stat[20 + 1]

                OWNship_longitude = float(SILS_stat[21 + 1])
                Ownship_latitude = float(SILS_stat[22 + 1])
                OWNship_altitude = float(SILS_stat[23 + 1])
                OWNship_heading = float(SILS_stat[24 + 1])
                OWNship_velocity = float(SILS_stat[25 + 1])
                OWNship_pos_x = float(SILS_stat[26 + 1])
                OWNship_pos_y = float(SILS_stat[27 + 1])
                OWNship_pos_z = float(SILS_stat[28 + 1])
                OWNship_roll = float(SILS_stat[29 + 1])
                OWNship_pitch = float(SILS_stat[30 + 1])
                OWNship_yaw = float(SILS_stat[31 + 1])
                OWNship_vertical_rate = float(SILS_stat[32 + 1])

                INTruder_longitude = float(SILS_stat[33 + 1])
                INTruderlatitude = float(SILS_stat[34 + 1])
                INTruderaltitude = float(SILS_stat[35 + 1])
                INTruder_heading = float(SILS_stat[36 + 1])
                INTruder_velocity = float(SILS_stat[37 + 1])
                INTruder_pos_x = float(SILS_stat[38 + 1])
                INTruder_pos_y = float(SILS_stat[39 + 1])
                INTruder_pos_z = float(SILS_stat[40 + 1])
                INTruder_roll = float(SILS_stat[41 + 1])
                INTruder_pitch = float(SILS_stat[42 + 1])
                INTruder_yaw = float(SILS_stat[43 + 1])
                INTruder_vertical_rate = float(SILS_stat[44 + 1])

                gamma = np.arcsin(OWNship_vertical_rate /
                                  max(1, OWNship_velocity, key=abs))
                theta = OWNship_pitch * Deg2Rad
                r = ATAR_intruder_relative_distance
                vc = -ATAR_intruder_closing_velocity
                elev = ATAR_intruder_elevation * Deg2Rad
                azim = -ATAR_intruder_bearing * Deg2Rad
                los = -elev

                if ATAR_update_couter != ATAR_update_couter_p:
                    # filtered LOS rate, bw=10, zeta=0.8, dt=0.01
                    dlos = 4.61893764e-01 * (los - los_p2) + \
                           1.84295612 * dlos_p - 0.852194 * dlos_p2
                    # filtered azim rate, bw=10, zeta=0.8, dt=0.01
                    daz = 4.61893764e-01 * (azim - azim_p2) + \
                          1.84295612 * daz_p - 0.852194 * daz_p2

                    los_p2 = los_p
                    los_p = los
                    dlos_p2 = dlos_p
                    dlos_p = dlos
                    azim_p2 = azim_p
                    azim_p = azim
                    daz_p2 = daz_p
                    daz_p = daz
                    ATAR_update_couter_p = ATAR_update_couter
                    zem_v = r * r / max(1, vc) * dlos
                    zem_h = r * r / max(1, vc) * daz
                    crm_v = r * los

                if 1500 > ATAR_intruder_relative_distance > dist_sep and abs(elev) < 40 * Deg2Rad and abs(
                        azim) < 40 * Deg2Rad:

                    thrustCMD = 150 + 100 * (51.44 - OWNship_velocity)
                    thrustCMD = int(np.clip(thrustCMD, 0, 255))

                    input_dat = torch.tensor(((np.array([r, vc, los, daz, dlos])
                                               - mean) / std).astype(np.float32)).cuda()
                    output = tr_model(input_dat.view(-1, 5))
                    _, predicted = torch.max(output, 1)
                    if predicted[0] == 0:
                        hdot_cmd = 0.0
                    if predicted[0] == 1:
                        hdot_cmd = -10.0
                    if predicted[0] == 2:
                        hdot_cmd = 10.0

                    ac = K_alt * (hdot_cmd - OWNship_vertical_rate) + \
                         g / (np.cos(gamma)) - 7.0
                    ac = np.clip(ac, -30, 30)
                    setCMD(client_ip, client_port, 0, 100, ac + 100, thrustCMD, True)

                else:
                    thrustCMD = 150 + 100 * (51.44 - OWNship_velocity)
                    thrustCMD = int(np.clip(thrustCMD, 0, 255))
                    hdot_cmd = 0
                    ac = K_alt * (hdot_cmd - OWNship_vertical_rate) + \
                         g / (np.cos(gamma)) - 7.0
                    ac = np.clip(ac, -30, 30)
                    setCMD(0, 100, ac + 100, thrustCMD, True)

                results.append([ATAR_intruder_bearing, ATAR_intruder_elevation, ATAR_intruder_relative_distance,
                                ATAR_intruder_closing_velocity, ATAR_intruder_detection_flag, OWNship_longitude,
                                Ownship_latitude, OWNship_altitude,
                                OWNship_heading, OWNship_velocity, OWNship_pos_x, OWNship_pos_y, OWNship_pos_z,
                                OWNship_roll, OWNship_pitch, OWNship_yaw,
                                OWNship_vertical_rate, gamma, ac, hdot_cmd, zem_v, zem_h, crm_v, daz, dlos])

                print("hdot_cmd:", hdot_cmd)
            except socket.timeout:
                detection_flag = 0
                print("timeout 1")
                continue
