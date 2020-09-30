import numpy as np
import cv2
from pathlib import Path
import sys
import pickle
import gzip
import pandas as pd
import json
from scipy.interpolate import interp1d

def marker_corners(x,y,l):
    c0 = [x,y]
    c1 = [x + l, y]
    c2 = [x + l, y + l]
    c3 = [x, y + l]
    return [c0, c1, c2, c3]

def find_targets(basedir, maxframes=np.inf):
    videos = Path(basedir).glob("*/world.mp4")
    video = next(videos)
    assert next(videos, None) is None
    videotime = video.parent / "world_timestamps.npy"
    assert videotime.exists()

    markers = Path(str(video) + ".markers.npy")
    assert markers.exists()

    pupilinfo = dict(l.split(',', 1) for l in open(video.parent / 'info.csv'))
    unix_offset = float(pupilinfo['Start Time (System)']) - float(pupilinfo['Start Time (Synced)'])

    markers = np.load(markers, allow_pickle=True)

    video = cv2.VideoCapture(str(video))
    videotimes = np.load(videotime)

    camera_spec = "pupil.fisheye.crop.720p.pickle.gz"
    camera = pickle.load(gzip.open(camera_spec, 'rb'), encoding='bytes')
    image_resolution = camera[b'resolution']
    
    rmap = camera[b'rect_map']
    rect_camera_matrix = camera[b'rect_camera_matrix']
    
    downscale = 4
    h, w = 1080, 1920
    h //= downscale
    w //= downscale
    width, height = w, h
    size = 0.1
    x = h*size*(1/7.)
    y = h*size*(1/7.)
    l = (5./7.) * h*size
    id0 = marker_corners(x, h - y - l, l)
    id1 = marker_corners(w - x - l, h - y - l, l)
    id2 = marker_corners(x,y,l)
    id3 = marker_corners(w - x - l, y, l)
    marker_dict = {"0": id0, "1": id1, "2": id2, "3": id3}
    
    template = cv2.imread('targettemplate.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.resize(template, tuple(np.array(template.shape)//downscale))
    mh, mw = template.shape



    for i, marker in enumerate(markers):
        if len(marker['markers']) == 0: continue

        frame_i = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if i != frame_i:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            frame_i = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        
        assert frame_i == i
        ret, oframe = video.read()
        assert ret

        vts = video.get(cv2.CAP_PROP_POS_MSEC)*1000
        
        frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY)

        frame = cv2.remap(frame, rmap[0], rmap[1], cv2.INTER_LINEAR)
        
        #from square_marker_detect import draw_markers
        #draw_markers(frame, marker['markers'])
        
        marker_positions = []
        screen_positions = []

        for m in marker['markers']:
            world = marker_dict[str(m['id'])]
            for w, s in zip(world, m['verts']):
                marker_positions.append(w)
                screen_positions.append(s)
        
        marker_positions = np.array(marker_positions, dtype=np.float32).reshape(-1, 1, 2)
        screen_positions = np.array(screen_positions, dtype=np.float32).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(screen_positions, marker_positions, 0)
        
        screen = cv2.warpPerspective(frame, M, (width, height))
        
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        x = max_loc[0] + mw/2
        y = max_loc[1] + mh/2
        
        x = x/width
        y = y/height


        
        yield {
            'x': x,
            'y': y,
            'pupil_time': videotimes[i],
            'unix_time': videotimes[i] + unix_offset,
            'video_frame': i,
            'video_time': vts,
            'match_correlation': max_val
        }


        #screen = cv2.circle(screen, (x, y), radius=5, color=(255, 0, 0), thickness=1)
        #
        #cv2.imshow("screen", screen[::-1])
        #cv2.imshow("match", (result[::-1] + 1)/2)
        #cv2.imshow("template", template)
        #cv2.waitKey(0)

        
    """
    for i, (vts, frame) in enumerate(frames()):
        ret, frame = video.read()
        if not ret: break


        ts = videotimes[i]

        marker = markers[i-1]
        
        print(i)
        if len(marker['markers']) == 0: continue
        for m in marker['markers']:
            print(m)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    """

def sync_session(basedir):
    import matplotlib.pyplot as plt
    logs = Path(basedir).glob("*/trajlog.jsons")
    log = next(logs)
    assert next(logs, None) is None
    rows = (json.loads(l) for l in open(log))
    ldata = []
    for row in rows:
        try:
            target = row['data']['pdBase']['platform']
        except KeyError:
            continue

        x = target['elements']["12"]
        y = target['elements']["13"]
        #if 'target' not in row['data']: continue
        ldata.append({
            'recv_ts_mono': row['recv_ts_mono'],
            'recv_ts_unix': row['recv_ts'],
            'ts_unix': row['time'],
            'x': x,
            'y': y
            })

    ldata = pd.DataFrame.from_records(ldata)
    
    ldata.y += 1.0
    ldata.y /= 2.0

    ldata.x /= 16/9
    ldata.x += 1.0
    ldata.x /= 2.0

    targets = find_targets(basedir)
    tdata = []
    for target in targets:
        if target['match_correlation'] < 0.7: continue
        tdata.append(target)
    
    tdata = pd.DataFrame.from_records(tdata)
    
    
    """
    plt.plot(ldata.ts_unix, ldata.y)
    plt.plot(tdata.unix_time, tdata.y, '.')
    #plt.plot(tdata.pupil_time, tdata.y, '.')
    plt.xlim(*tdata.unix_time.iloc[[0,-1]])
    """

    linterp = interp1d(ldata.ts_unix.values, ldata[['x', 'y']].values, axis=0)
    ttime = tdata.unix_time.values
    tpos = tdata[['x', 'y']].values

    lags = np.linspace(-0.5, 0.5, 1000)
    errs = []

    for lag in lags:
        lpos = linterp(tdata.unix_time + lag) - tpos
        err = np.mean(np.linalg.norm(lpos, axis=1))
        errs.append(err)

    plt.plot(lags, errs)

    print("Best lag", lags[np.argmin(errs)])

    plt.show()

if __name__ == '__main__':
    sync_session(sys.argv[1])
