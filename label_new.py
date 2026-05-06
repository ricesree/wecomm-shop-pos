"""
Labeler for "all new" images — same UI as label.py
----------------------------------------------------
Opens every folder inside "all new/" and lets you draw bounding boxes.
Saves YOLO .txt label files alongside each image (same folder).
prepare_categories.py will automatically use these real boxes.

Controls:
  B     = Rectangle box  (drag)
  P     = Pen / freehand
  O     = Sloped box
  ENTER = save & next image
  Z     = undo last box
  R     = reset to auto-box
  S     = skip image
  Q     = quit
"""

import cv2, numpy as np, os

NEW_DATA_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\all new"

COLORS = [
    (0,220,0),(0,180,255),(255,100,0),
    (180,0,255),(0,255,180),(255,220,0),(0,255,255),
]

mode       = "auto"
holding    = False
cur_sx = cur_sy = cur_ex = cur_ey = -1
pen_pts    = []
boxes      = []
obb_phase  = 0
obb_p1     = None
obb_p2     = None
obb_mouse  = (0, 0)

def obb_corners(p1, p2, mouse):
    dx,dy  = p2[0]-p1[0], p2[1]-p1[1]
    length = max(np.hypot(dx,dy), 1)
    ux,uy  = dx/length, dy/length
    px,py  = -uy, ux
    mx,my  = mouse[0]-p1[0], mouse[1]-p1[1]
    w      = mx*px + my*py
    return [(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),
            (int(p2[0]+w*px),int(p2[1]+w*py)),(int(p1[0]+w*px),int(p1[1]+w*py))]

def mouse_cb(event, x, y, flags, param):
    global holding,cur_sx,cur_sy,cur_ex,cur_ey,pen_pts
    global obb_phase,obb_p1,obb_p2,obb_mouse
    obb_mouse = (x,y)
    if mode == "box":
        if event==cv2.EVENT_LBUTTONDOWN:
            holding=True; cur_sx=cur_ex=x; cur_sy=cur_ey=y; pen_pts=[]
        elif event==cv2.EVENT_MOUSEMOVE and holding:
            cur_ex,cur_ey=x,y
        elif event==cv2.EVENT_LBUTTONUP and holding:
            holding=False; cur_ex,cur_ey=x,y
            if abs(cur_ex-cur_sx)>5 and abs(cur_ey-cur_sy)>5:
                x1,y1=min(cur_sx,cur_ex),min(cur_sy,cur_ey)
                x2,y2=max(cur_sx,cur_ex),max(cur_sy,cur_ey)
                boxes.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            cur_sx=cur_sy=cur_ex=cur_ey=-1
    elif mode == "pen":
        if event==cv2.EVENT_LBUTTONDOWN:
            holding=True; pen_pts=[(x,y)]
        elif event==cv2.EVENT_MOUSEMOVE and holding:
            pen_pts.append((x,y))
        elif event==cv2.EVENT_LBUTTONUP and holding:
            holding=False
            if len(pen_pts)>2:
                xs=[p[0] for p in pen_pts]; ys=[p[1] for p in pen_pts]
                boxes.append([(min(xs),min(ys)),(max(xs),min(ys)),
                               (max(xs),max(ys)),(min(xs),max(ys))])
            pen_pts=[]
    elif mode == "obb":
        if event==cv2.EVENT_LBUTTONDOWN:
            if obb_phase==0:
                obb_phase=1; obb_p1=(x,y); obb_p2=(x,y)
            elif obb_phase==2:
                boxes.append(obb_corners(obb_p1,obb_p2,(x,y)))
                obb_phase=0; obb_p1=obb_p2=None
        elif event==cv2.EVENT_MOUSEMOVE:
            if obb_phase==1: obb_p2=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and obb_phase==1:
            if obb_p1 and np.hypot(x-obb_p1[0],y-obb_p1[1])>5:
                obb_p2=(x,y); obb_phase=2
            else:
                obb_phase=0

def auto_bbox(img):
    h0,w0=img.shape[:2]; scale=0.25
    small=cv2.resize(img,(int(w0*scale),int(h0*scale))); h,w=small.shape[:2]
    mx,my=w//6,h//6; rect=(mx,my,w-2*mx,h-2*my)
    mask=np.zeros((h,w),np.uint8)
    bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(small,mask,rect,bgd,fgd,5,cv2.GC_INIT_WITH_RECT)
        fg=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
        k=np.ones((10,10),np.uint8)
        fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,k,iterations=2)
        fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,k,iterations=1)
        cnts,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            large=[c for c in cnts if cv2.contourArea(c)>w*h*0.015] or [max(cnts,key=cv2.contourArea)]
            pts=np.concatenate(large); x,y,bw,bh=cv2.boundingRect(pts)
            s=1/scale; pad=40
            x=max(0,int(x*s)-pad); y=max(0,int(y*s)-pad)
            bw=min(w0-x,int(bw*s)+2*pad); bh=min(h0-y,int(bh*s)+2*pad)
            return x,y,x+bw,y+bh
    except Exception:
        pass
    px,py=int(w0*0.175),int(h0*0.175)
    return px,py,w0-px,h0-py

def resize_to_fit(img,max_w=1050,max_h=700):
    h,w=img.shape[:2]; sc=min(max_w/w,max_h/h,1.0)
    if sc<1.0: img=cv2.resize(img,(int(w*sc),int(h*sc)))
    return img,sc

def build_frame(disp,folder,fname,idx,total):
    vis=disp.copy()
    for i,corners in enumerate(boxes):
        col=COLORS[i%len(COLORS)]
        cv2.polylines(vis,[np.array(corners,np.int32)],True,col,2)
        cv2.putText(vis,f"#{i+1}",corners[0],cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
    if mode=="box" and holding and cur_sx!=-1:
        cv2.rectangle(vis,(min(cur_sx,cur_ex),min(cur_sy,cur_ey)),
                          (max(cur_sx,cur_ex),max(cur_sy,cur_ey)),(180,180,180),1)
    if mode=="pen" and holding and len(pen_pts)>1:
        cv2.polylines(vis,[np.array(pen_pts,np.int32)],False,(180,180,180),1)
    if mode=="obb" and obb_phase>=1 and obb_p1 and obb_p2:
        if obb_phase==1:
            cv2.line(vis,obb_p1,obb_p2,(200,200,200),1)
        elif obb_phase==2:
            cv2.polylines(vis,[np.array(obb_corners(obb_p1,obb_p2,obb_mouse),np.int32)],True,(0,200,255),2)

    sb_w=240; dh,dw=vis.shape[:2]
    sb=np.zeros((dh,sb_w,3),np.uint8)
    def t(s,y,col=(200,200,200),sc=0.47):
        cv2.putText(sb,s,(8,y),cv2.FONT_HERSHEY_SIMPLEX,sc,col,1)
    t(folder[:28],18,(100,200,100),0.46)
    t(fname[:28],  40,(255,255,255),0.42)
    t(f"Image {idx}/{total}",64,(150,150,150))
    t(f"Boxes: {len(boxes)}",88,(255,220,80),0.52)
    t("── DRAW MODE ─────",115,(70,70,70))
    t("[B] Rectangle",     137,(255,140,0)  if mode=="box"  else (55,55,55))
    t("[P] Freehand pen",  158,(0,120,255)  if mode=="pen"  else (55,55,55))
    t("[O] Sloped box",    179,(0,220,220)  if mode=="obb"  else (55,55,55))
    t("── ACTIONS ───────",210,(70,70,70))
    t("ENTER  save & next",232,(255,255,255))
    t("Z      undo last",  254,(200,200,200))
    t("R      reset auto", 276,(200,200,200))
    t("S      skip",       298,(200,200,200))
    t("Q      quit",       320,(200,200,200))
    return np.hstack([vis,sb])

def corners_to_yolo(corners,scale,ow,oh):
    xs=[int(c[0]/scale) for c in corners]; ys=[int(c[1]/scale) for c in corners]
    x1,y1=max(0,min(xs)),max(0,min(ys))
    x2,y2=min(ow,max(xs)),min(oh,max(ys))
    xc=(x1+x2)/2/ow; yc=(y1+y2)/2/oh
    bw=(x2-x1)/ow;   bh=(y2-y1)/oh
    return xc,yc,bw,bh

def main():
    global mode,holding,cur_sx,cur_sy,cur_ex,cur_ey,pen_pts,boxes
    global obb_phase,obb_p1,obb_p2,obb_mouse

    folders = sorted([d for d in os.listdir(NEW_DATA_DIR)
                      if os.path.isdir(os.path.join(NEW_DATA_DIR, d))])
    print(f"Found {len(folders)} folders in 'all new/'")
    for f in folders:
        print(f"  {f}")

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Labeler", mouse_cb)

    saved = quit_all = 0

    for folder in folders:
        if quit_all: break
        folder_path = os.path.join(NEW_DATA_DIR, folder)
        imgs = sorted([f for f in os.listdir(folder_path)
                       if f.lower().endswith((".jpg",".jpeg",".png"))])
        print(f"\n[{folder}] — {len(imgs)} images")

        for idx, fname in enumerate(imgs, 1):
            if quit_all: break
            img_path = os.path.join(folder_path, fname)
            lbl_path = os.path.join(folder_path, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(lbl_path):
                saved += 1; continue

            orig = cv2.imread(img_path)
            if orig is None: continue
            disp, scale = resize_to_fit(orig)
            oh, ow = orig.shape[:2]

            ax1,ay1,ax2,ay2 = auto_bbox(orig)
            auto_corners = [(int(ax1*scale),int(ay1*scale)),(int(ax2*scale),int(ay1*scale)),
                            (int(ax2*scale),int(ay2*scale)),(int(ax1*scale),int(ay2*scale))]

            mode="auto"; holding=False; pen_pts=[]
            boxes=[auto_corners[:]]; cur_sx=cur_sy=cur_ex=cur_ey=-1
            obb_phase=0; obb_p1=obb_p2=None

            while True:
                cv2.imshow("Labeler", build_frame(disp,folder,fname,idx,len(imgs)))
                key = cv2.waitKey(20) & 0xFF

                if key in (13,10):          # ENTER — save
                    if boxes:
                        lines = []
                        for corners in boxes:
                            xc,yc,bw,bh = corners_to_yolo(corners,scale,ow,oh)
                            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                        with open(lbl_path,"w") as f:
                            f.write("\n".join(lines)+"\n")
                        saved += 1
                        print(f"  ✓ {fname}  ({len(boxes)} box)")
                    break
                elif key in (ord('b'),ord('B')): mode="box";  pen_pts=[];holding=False;obb_phase=0;cur_sx=cur_sy=cur_ex=cur_ey=-1
                elif key in (ord('p'),ord('P')): mode="pen";  pen_pts=[];holding=False;obb_phase=0
                elif key in (ord('o'),ord('O')): mode="obb";  pen_pts=[];holding=False;obb_phase=0
                elif key in (ord('z'),ord('Z')):
                    if boxes: boxes.pop()
                elif key in (ord('r'),ord('R')):
                    mode="auto"; pen_pts=[];holding=False;obb_phase=0
                    boxes=[auto_corners[:]]; cur_sx=cur_sy=cur_ex=cur_ey=-1
                elif key in (ord('s'),ord('S')):
                    print(f"  – skipped {fname}"); break
                elif key in (ord('q'),ord('Q')):
                    quit_all=True; break

    cv2.destroyAllWindows()
    print(f"\nDone! {saved} images labeled.")

if __name__ == "__main__":
    main()
