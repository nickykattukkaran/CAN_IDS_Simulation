sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
ip link show vcan0
candump vcan0
cansend vcan0 123#deadbeef
cansend vcan0 123#1122334455667788
cangen vcan0
send:
python3 send_can.py -i vcan0 -c can_combine_simulation.csv
receiver:
python3 receiver_process.py -i vcan0 -o receive_message1.csv -d 10
