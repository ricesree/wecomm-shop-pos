"""
Multi-Box Labeler  —  Box / Pen / Sloped-Box per image
--------------------------------------------------------
Controls:
  B           = Rectangle box   (drag)
  P           = Pen / freehand  (hold & draw)
  O           = Sloped box      (drag first edge → move mouse for width → click)
  ENTER       = save ALL boxes → next image
  Z           = undo last box
  R           = reset to machine auto-box
  S           = skip image
  Q           = quit
"""

import cv2
import numpy as np
import os

DATASET_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
LABELS_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"

COLORS = [
    (0,220,0),(0,180,255),(255,100,0),
    (180,0,255),(0,255,180),(255,220,0),(0,255,255),
]

# ── state ─────────────────────────────────────────────────────────────────────
mode       = "auto"
holding    = False
cur_sx = cur_sy = cur_ex = cur_ey = -1
pen_pts    = []
boxes      = []          # confirmed boxes: each is list of (x,y) corners

# sloped-box phases
obb_phase  = 0           # 0=idle  1=dragging edge  2=setting width
obb_p1     = None
obb_p2     = None
obb_mouse  = (0, 0)

def obb_corners(p1, p2, mouse):
    dx,dy  = p2[0]-p1[0], p2[1]-p1[1]
    length = max(np.hypot(dx,dy), 1)
    ux,uy  = dx/length, dy/length   # along edge
    px,py  = -uy, ux                # perpendicular
    mx,my  = mouse[0]-p1[0], mouse[1]-p1[1]
    w      = mx*px + my*py
    c1 = (int(p1[0]),           int(p1[1]))
    c2 = (int(p2[0]),           int(p2[1]))
    c3 = (int(p2[0]+w*px),      int(p2[1]+w*py))
    c4 = (int(p1[0]+w*px),      int(p1[1]+w*py))
    return [c1,c2,c3,c4]

def mouse_cb(event, x, y, flags, param):
    global holding,cur_sx,cur_sy,cur_ex,cur_ey,pen_pts
    global obb_phase,obb_p1,obb_p2,obb_mouse

    obb_mouse = (x,y)

    # ── BOX ──
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

    # ── PEN ──
    elif mode == "pen":
        if event==cv2.EVENT_LBUTTONDOWN:
            holding=True; pen_pts=[(x,y)]
        elif event==cv2.EVENT_MOUSEMOVE and holding:
            pen_pts.append((x,y))
        elif event==cv2.EVENT_LBUTTONUP and holding:
            holding=False
            if len(pen_pts)>2:
                xs=[p[0] for p in pen_pts]; ys=[p[1] for p in pen_pts]
                x1,y1=min(xs),min(ys); x2,y2=max(xs),max(ys)
                boxes.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            pen_pts=[]

    # ── SLOPED BOX ──
    elif mode == "obb":
        if event==cv2.EVENT_LBUTTONDOWN:
            if obb_phase==0:
                obb_phase=1; obb_p1=(x,y); obb_p2=(x,y)
            elif obb_phase==2:
                # confirm width → add box
                corners=obb_corners(obb_p1,obb_p2,(x,y))
                boxes.append(corners)
                obb_phase=0; obb_p1=obb_p2=None
        elif event==cv2.EVENT_MOUSEMOVE:
            if obb_phase==1:
                obb_p2=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and obb_phase==1:
            if obb_p1 and np.hypot(x-obb_p1[0],y-obb_p1[1])>5:
                obb_p2=(x,y); obb_phase=2   # edge set → now set width
            else:
                obb_phase=0

# ── auto bbox ─────────────────────────────────────────────────────────────────
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
            large=[c for c in cnts if cv2.contourArea(c)>w*h*0.015] or \
                  [max(cnts,key=cv2.contourArea)]
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

def build_frame(disp,cls,idx,total):
    vis=disp.copy()

    # confirmed boxes
    for i,corners in enumerate(boxes):
        col=COLORS[i%len(COLORS)]
        pts=np.array(corners,np.int32)
        cv2.polylines(vis,[pts],True,col,2)
        cv2.putText(vis,f"#{i+1}",corners[0],cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)

    # in-progress previews
    if mode=="box" and holding and cur_sx!=-1:
        cv2.rectangle(vis,(min(cur_sx,cur_ex),min(cur_sy,cur_ey)),
                          (max(cur_sx,cur_ex),max(cur_sy,cur_ey)),(180,180,180),1)
    if mode=="pen" and holding and len(pen_pts)>1:
        cv2.polylines(vis,[np.array(pen_pts,np.int32)],False,(180,180,180),1)
    if mode=="obb" and obb_phase>=1 and obb_p1 and obb_p2:
        if obb_phase==1:
            cv2.line(vis,obb_p1,obb_p2,(200,200,200),1)
        elif obb_phase==2:
            corners=obb_corners(obb_p1,obb_p2,obb_mouse)
            cv2.polylines(vis,[np.array(corners,np.int32)],True,(0,200,255),2)

    # sidebar
    sb_w=220; dh,dw=vis.shape[:2]
    sb=np.zeros((dh,sb_w,3),np.uint8)

    def t(s,y,col=(200,200,200),sc=0.47):
        cv2.putText(sb,s,(8,y),cv2.FONT_HERSHEY_SIMPLEX,sc,col,1)

    t(f"Class:",18,(100,200,100),0.50)
    t(f" {cls}",40,(255,255,255),0.48)
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

    # tips
    y=365
    if mode=="obb":
        if obb_phase==0:   t("Click+drag 1st edge",y,(0,220,220))
        elif obb_phase==1: t("Drag to set edge...",y,(0,220,220))
        elif obb_phase==2:
            t("Move mouse→ set",y,(0,255,180))
            t("width, then CLICK",y+22,(0,255,180))
    elif mode=="pen":
        if not holding: t("Hold & draw around",y,(0,120,255))
        else:           t("Keep drawing...",y,(0,255,150))
    elif mode=="box":
        if not holding: t("Click & drag box",y,(255,140,0))

    return np.hstack([vis,sb])

def corners_to_yolo(corners,scale,ow,oh):
    xs=[int(c[0]/scale) for c in corners]
    ys=[int(c[1]/scale) for c in corners]
    x1,y1=max(0,min(xs)),max(0,min(ys))
    x2,y2=min(ow,max(xs)),min(oh,max(ys))
    xc=(x1+x2)/2/ow; yc=(y1+y2)/2/oh
    bw=(x2-x1)/ow;   bh=(y2-y1)/oh
    return xc,yc,bw,bh

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global mode,holding,cur_sx,cur_sy,cur_ex,cur_ey,pen_pts,boxes
    global obb_phase,obb_p1,obb_p2,obb_mouse

    os.makedirs(LABELS_DIR,exist_ok=True)
    classes=sorted([d for d in os.listdir(DATASET_DIR)
                    if os.path.isdir(os.path.join(DATASET_DIR,d))])
    cv2.namedWindow("Labeler",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Labeler",mouse_cb)

    saved=0; quit_all=False

    for class_id,cls in enumerate(classes):
        if quit_all: break
        cls_dir=os.path.join(DATASET_DIR,cls)
        lbl_dir=os.path.join(LABELS_DIR,cls)
        os.makedirs(lbl_dir,exist_ok=True)
        imgs=sorted([f for f in os.listdir(cls_dir)
                     if f.lower().endswith((".jpg",".jpeg",".png"))])

        for idx,fname in enumerate(imgs,1):
            if quit_all: break
            lbl_path=os.path.join(lbl_dir,os.path.splitext(fname)[0]+".txt")
            if os.path.exists(lbl_path): saved+=1; continue

            orig=cv2.imread(os.path.join(cls_dir,fname))
            if orig is None: continue
            disp,scale=resize_to_fit(orig); oh,ow=orig.shape[:2]

            ax1,ay1,ax2,ay2=auto_bbox(orig)
            dax1=int(ax1*scale); day1=int(ay1*scale)
            dax2=int(ax2*scale); day2=int(ay2*scale)
            auto_corners=[(dax1,day1),(dax2,day1),(dax2,day2),(dax1,day2)]

            mode="auto"; holding=False; pen_pts=[]
            boxes=[auto_corners[:]];
            cur_sx=cur_sy=cur_ex=cur_ey=-1
            obb_phase=0; obb_p1=obb_p2=None

            while True:
                frame=build_frame(disp,cls,idx,len(imgs))
                cv2.imshow("Labeler",frame)
                key=cv2.waitKey(20)&0xFF

                if key in (13,10):                       # ENTER
                    if boxes:
                        lines=[]
                        for corners in boxes:
                            xc,yc,bw,bh=corners_to_yolo(corners,scale,ow,oh)
                            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                        with open(lbl_path,"w") as f:
                            f.write("\n".join(lines)+"\n")
                        saved+=1
                        print(f"  Saved {len(boxes)} box(es)  [{cls}] {fname}")
                    break

                elif key in (ord('b'),ord('B')):
                    mode="box"; pen_pts=[]; holding=False; obb_phase=0
                    cur_sx=cur_sy=cur_ex=cur_ey=-1
                elif key in (ord('p'),ord('P')):
                    mode="pen"; pen_pts=[]; holding=False; obb_phase=0
                    cur_sx=cur_sy=cur_ex=cur_ey=-1
                elif key in (ord('o'),ord('O')):
                    mode="obb"; pen_pts=[]; holding=False; obb_phase=0
                    cur_sx=cur_sy=cur_ex=cur_ey=-1
                elif key in (ord('z'),ord('Z')):
                    if boxes: boxes.pop()
                elif key in (ord('r'),ord('R')):
                    mode="auto"; pen_pts=[]; holding=False; obb_phase=0
                    boxes=[auto_corners[:]]; cur_sx=cur_sy=cur_ex=cur_ey=-1
                elif key in (ord('s'),ord('S')):
                    print(f"  Skipped [{cls}] {fname}"); break
                elif key in (ord('q'),ord('Q')):
                    quit_all=True; break

    cv2.destroyAllWindows()
    print(f"\nDone! {saved} images labeled.\nLabels: {LABELS_DIR}")

if __name__=="__main__":
    main()
