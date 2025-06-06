sudo modprobe vcan<br>
sudo ip link add dev vcan0 type vcan<br>
sudo ip link set up vcan0<br>
ip link show vcan0<br>
candump vcan0<br>
cansend vcan0 123#deadbeef<br>
cansend vcan0 123#1122334455667788<br>
cangen vcan0<br>
<b>send:</b><br>
python3 send_can.py -i vcan0 -c can_combine_simulation.csv<br>
<b>receiver:</b><br>
python3 receiver_process.py -i vcan0 -o receive_message1.csv -d 10<br>
