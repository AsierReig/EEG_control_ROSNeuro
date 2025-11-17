import rosbag
bag = rosbag.Bag("/home/tartaria/catkin_ws/src/my_hero_bci/nodes/2025-11-12-09-57-13.bag")
for _, msg, _ in bag.read_messages(topics=['/neurodata']):
    print("Channels:", msg.eeg.info.nchannels)
    print("Samples:", msg.eeg.info.nsamples)
    print("Data length:", len(msg.eeg.data))
    print("Labels:", msg.eeg.info.labels)
    break
bag.close()
