from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/libero", episodes=[0])

print("fps:", ds.fps)
for k, v in ds.meta.features.items():
    print(f"{k:40s} {v['dtype']:8s} {v['shape']}")

a = ds.meta.stats["action"]
print("action min:", a["min"])
print("action max:", a["max"])
print("action mean:", a["mean"])

s = ds.meta.stats["observation.state"]
print("state min:", s["min"])
print("state max:", s["max"])

frame = ds[0]
print("task:", frame["task"])
print("first action:", frame["action"])