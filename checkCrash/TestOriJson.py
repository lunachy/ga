# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    TestOriJson
    Description :    测试JSON
    Author      :    zhaowen
    date        :    2019/5/31
____________________________________________________________________
    Change Activity:
                        2019/5/31:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'
import json

import numpy as np
import json
from matplotlib import patches, lines
import matplotlib.pyplot as plt

jsonpath = r"C:\Users\zhaowen\Desktop\0(11)\0\1-room-json.json"
json_data = "{\"houseId\":1023616,\"functionZones\":[{\"id\":48,\"positions\":[{\"sample\":false,\"main\":true,\"tag\":\"PrgMlrjz\",\"cid\":318,\"metaData\":{\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":3.141592653589793},\"location\":{\"x\":-936,\"y\":803,\"z\":0},\"scale\":{\"x\":1.0,\"y\":1.0,\"z\":1.0},\"size\":{\"dx\":1321,\"dy\":1490,\"dz\":1490},\"modelId\":0,\"skuId\":0,\"isCustom\":false,\"spuId\":0,\"mark\":0},\"relativeLocation\":{\"x\":0,\"y\":0,\"z\":0},\"relativeRotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"isDrModel\":false},{\"sample\":true,\"main\":false,\"tag\":\"J4m2VsgC\",\"cid\":320,\"metaData\":{\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":3.141592653589793},\"location\":{\"x\":-895,\"y\":-190,\"z\":0},\"scale\":{\"x\":1.0,\"y\":1.0,\"z\":1.0},\"size\":{\"dx\":1314,\"dy\":420,\"dz\":420},\"modelId\":0,\"skuId\":0,\"isCustom\":false,\"spuId\":0,\"mark\":0},\"relativeLocation\":{\"x\":-41,\"y\":993,\"z\":0},\"relativeRotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"isDrModel\":false},{\"sample\":true,\"main\":false,\"tag\":\"zaU3oNlM\",\"cid\":319,\"metaData\":{\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":3.141592653589793},\"location\":{\"x\":-70,\"y\":1413,\"z\":0},\"scale\":{\"x\":1.0,\"y\":1.0,\"z\":1.0},\"size\":{\"dx\":445,\"dy\":275,\"dz\":275},\"modelId\":0,\"skuId\":0,\"isCustom\":false,\"spuId\":0,\"mark\":0},\"relativeLocation\":{\"x\":-866,\"y\":-610,\"z\":0},\"relativeRotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"isDrModel\":false},{\"sample\":true,\"main\":false,\"tag\":\"5000QrD5\",\"cid\":319,\"metaData\":{\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":3.141592653589793},\"location\":{\"x\":-1846,\"y\":1400,\"z\":0},\"scale\":{\"x\":1.0,\"y\":1.0,\"z\":1.0},\"size\":{\"dx\":461,\"dy\":309,\"dz\":309},\"modelId\":0,\"skuId\":0,\"isCustom\":false,\"spuId\":0,\"mark\":0},\"relativeLocation\":{\"x\":910,\"y\":-597,\"z\":0},\"relativeRotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"isDrModel\":false}],\"bound\":{\"x\":-2077,\"y\":-400,\"dx\":2229,\"dy\":1955},\"center\":{\"x\":-936,\"y\":803,\"z\":0},\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":3.141592653589793},\"tag\":0},{\"id\":38,\"positions\":[{\"sample\":false,\"main\":true,\"tag\":\"Js4omjTN\",\"cid\":120,\"metaData\":{\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":1.5707963267948966},\"location\":{\"x\":968,\"y\":667,\"z\":0},\"scale\":{\"x\":1.0,\"y\":1.0,\"z\":1.0},\"size\":{\"dx\":1748,\"dy\":466,\"dz\":466},\"modelId\":0,\"skuId\":0,\"isCustom\":false,\"spuId\":0,\"mark\":0},\"relativeLocation\":{\"x\":0,\"y\":0,\"z\":0},\"relativeRotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"isDrModel\":false}],\"bound\":{\"x\":93,\"y\":433,\"dx\":1748,\"dy\":466},\"center\":{\"x\":968,\"y\":667,\"z\":0},\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":1.5707963267948966},\"tag\":1},{\"id\":54,\"positions\":[{\"sample\":false,\"main\":true,\"tag\":\"oEKYZgZL\",\"cid\":108,\"metaData\":{\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"location\":{\"x\":-1027,\"y\":-1352,\"z\":0},\"scale\":{\"x\":1.0,\"y\":1.0,\"z\":1.0},\"size\":{\"dx\":1483,\"dy\":391,\"dz\":391},\"modelId\":0,\"skuId\":0,\"isCustom\":false,\"spuId\":0,\"mark\":0},\"relativeLocation\":{\"x\":0,\"y\":0,\"z\":0},\"relativeRotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"isDrModel\":false}],\"bound\":{\"x\":-1769,\"y\":-1547,\"dx\":1483,\"dy\":391},\"center\":{\"x\":-1027,\"y\":-1352,\"z\":0},\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"tag\":2},{\"id\":99,\"positions\":[],\"bound\":{\"x\":0,\"y\":0,\"dx\":0,\"dy\":0},\"center\":{\"x\":0,\"y\":0,\"z\":0},\"rotate\":{\"xAxis\":0.0,\"yAxis\":0.0,\"zAxis\":0.0},\"tag\":3}],\"usageId\":2,\"roomName\":\"主卧\",\"walls\":[{\"scid\":16,\"wallPoints\":[{\"x\":1204,\"y\":-1549},{\"x\":2405,\"y\":-1549},{\"x\":1204,\"y\":-1750},{\"x\":2405,\"y\":-1750}]},{\"scid\":1,\"wallPoints\":[{\"x\":-2605,\"y\":-1549},{\"x\":1205,\"y\":-1549},{\"x\":-2605,\"y\":-1750},{\"x\":1205,\"y\":-1750}]},{\"scid\":1,\"wallPoints\":[{\"x\":-2605,\"y\":1750},{\"x\":-2404,\"y\":1750},{\"x\":-2605,\"y\":-1750},{\"x\":-2404,\"y\":-1750}]},{\"scid\":1,\"wallPoints\":[{\"x\":-2605,\"y\":1750},{\"x\":1215,\"y\":1750},{\"x\":-2605,\"y\":1549},{\"x\":1215,\"y\":1549}]},{\"scid\":13,\"wallPoints\":[{\"x\":2404,\"y\":-429},{\"x\":2605,\"y\":-429},{\"x\":2404,\"y\":-1600},{\"x\":2605,\"y\":-1600}]},{\"scid\":12,\"wallPoints\":[{\"x\":1214,\"y\":1650},{\"x\":1315,\"y\":1650},{\"x\":1214,\"y\":-430},{\"x\":1315,\"y\":-430}]},{\"scid\":12,\"wallPoints\":[{\"x\":1214,\"y\":-329},{\"x\":2505,\"y\":-329},{\"x\":1214,\"y\":-430},{\"x\":2505,\"y\":-430}]}],\"windows\":[{\"scid\":1,\"points\":[{\"x\":-2610,\"y\":-1508},{\"x\":-2610,\"y\":1504},{\"x\":-2408,\"y\":1504},{\"x\":-2408,\"y\":-1508}],\"type\":0,\"horizontalFlip\":false,\"verticalFlip\":false},{\"scid\":1,\"points\":[{\"x\":-2360,\"y\":1554},{\"x\":-2360,\"y\":1755},{\"x\":-1036,\"y\":1755},{\"x\":-1036,\"y\":1554}],\"type\":0,\"horizontalFlip\":false,\"verticalFlip\":false}],\"doors\":[{\"scid\":13,\"points\":[{\"x\":2404,\"y\":-1439},{\"x\":2404,\"y\":-538},{\"x\":2605,\"y\":-538},{\"x\":2605,\"y\":-1439}],\"type\":0,\"horizontalFlip\":true,\"verticalFlip\":true,\"rotate\":{\"zAxis\":179.99996185302734}}],\"areaPoints\":[{\"x\":1315,\"y\":-430},{\"x\":2404,\"y\":-430},{\"x\":2404,\"y\":-1549},{\"x\":1205,\"y\":-1549},{\"x\":1204,\"y\":-1549},{\"x\":-2404,\"y\":-1549},{\"x\":-2404,\"y\":1549},{\"x\":1214,\"y\":1549},{\"x\":1214,\"y\":-329},{\"x\":1214,\"y\":-430},{\"x\":1315,\"y\":-430}],\"roomId\":3129002}";
json_data = json.loads(json_data)
data = json_data
debug = True
show=["cid"]
fig, ax = plt.subplots()
for zone in data["functionZones"]:
    k = "bound"

    if debug:
        print("*" * 20)

    pos = zone[k]
    # if int((pos["axisRotate"]["zAxis"] / np.pi) * 180) in [90, -90, 270]:
    #     dx = pos["size"]["dy"]
    #     dy = pos["size"]["dx"]
    # else:
    #     dx = pos["size"]["dx"]
    #     dy = pos["size"]["dy"]
    # x = pos["location"]["x"] - 0.5 * dx
    # y = pos["location"]["y"] - 0.5 * dy
    x = pos["x"] +3200
    y = pos["y"]+3200
    dx = pos["dx"]
    dy = pos["dy"]
    print((x, y), dx, dy)
    p = patches.Rectangle((x, y), dx, dy, linewidth=2,
                          alpha=1.0, linestyle="dashed",
                          edgecolor="y", facecolor='none')
    ax.add_patch(p)
    ax.text(x=x, y=y, s="  ")

plt.show()
