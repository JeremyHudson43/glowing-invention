
import random, json, os, time
print('Running attempt 2 – change type: focal_loss')
# Dummy training (simulate work)
time.sleep(0.5)
base_f1 = random.uniform(0.1, 0.3)
new_f1 = base_f1 + random.uniform(-0.05, 0.05)
improved = new_f1 > base_f1
print(json.dumps({ 'base_f1': base_f1, 'new_f1': new_f1, 'improved': improved }))
