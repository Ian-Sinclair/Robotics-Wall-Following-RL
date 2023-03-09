#!/usr/bin/env python3


scan_r = list(range(20))

deg = 360 / len(scan_r)
print(deg)
scan_range = []
for i in range(360) :
    scan_range += [ scan_r[ int( i/deg ) ] ]

print(len(scan_range))