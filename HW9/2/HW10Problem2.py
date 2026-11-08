"""HW9 Problem 2: Q4/Q8 Jacobian exploration.
Uses NumPy + Python standard library only (no Matplotlib/GUI backend).
Writes SVG plots beside this script; SVGs open in a web browser on Windows.
"""
import os
import sys
import traceback
import webbrowser
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent


def shape_functions(xi, eta, n):
    if n == 4:
        return 0.25 * np.array([
            (1-xi)*(1-eta), (1+xi)*(1-eta),
            (1+xi)*(1+eta), (1-xi)*(1+eta)])
    if n == 8:
        return np.array([
            -0.25*(1-xi)*(1-eta)*(1+xi+eta),
             0.25*(1+xi)*(1-eta)*(xi-eta-1),
             0.25*(1+xi)*(1+eta)*(xi+eta-1),
             0.25*(1-xi)*(1+eta)*(-xi+eta-1),
             0.5*(1-xi**2)*(1-eta),
             0.5*(1+xi)*(1-eta**2),
             0.5*(1-xi**2)*(1+eta),
             0.5*(1-xi)*(1-eta**2)])
    raise ValueError('Only Q4 and Q8 elements are supported.')


def shape_function_gradients(xi, eta, n):
    if n == 4:
        return 0.25*np.array([
            [-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
            [(1+eta), (1+xi)], [-(1+eta), (1-xi)]]).T
    if n == 8:
        return np.array([
            [-0.25*(-1+eta)*(2*xi+eta), -0.25*(-1+xi)*(xi+2*eta)],
            [ 0.25*(-1+eta)*(-2*xi+eta), 0.25*(1+xi)*(-xi+2*eta)],
            [ 0.25*(1+eta)*(2*xi+eta), 0.25*(1+xi)*(xi+2*eta)],
            [-0.25*(1+eta)*(-2*xi+eta), -0.25*(-1+xi)*(-xi+2*eta)],
            [xi*(-1+eta), 0.5*(1+xi)*(-1+xi)],
            [-0.5*(1+eta)*(-1+eta), -eta*(1+xi)],
            [-xi*(1+eta), -0.5*(1+xi)*(-1+xi)],
            [0.5*(1+eta)*(-1+eta), eta*(-1+xi)]]).T
    raise ValueError('Only Q4 and Q8 elements are supported.')


def jacobian(xi, eta, xy):
    """Build the 2x2 Jacobian using scalar sums (no BLAS/matrix multiply)."""
    grads = shape_function_gradients(xi, eta, len(xy))
    dx_dxi = 0.0
    dy_dxi = 0.0
    dx_deta = 0.0
    dy_deta = 0.0
    for i in range(len(xy)):
        dx_dxi += float(grads[0, i]) * float(xy[i, 0])
        dy_dxi += float(grads[0, i]) * float(xy[i, 1])
        dx_deta += float(grads[1, i]) * float(xy[i, 0])
        dy_deta += float(grads[1, i]) * float(xy[i, 1])
    return ((dx_dxi, dy_dxi), (dx_deta, dy_deta))


def map_to_physical(shape, xy):
    """Map a parent point without NumPy matrix multiplication."""
    x = 0.0
    y = 0.0
    for i in range(len(xy)):
        x += float(shape[i]) * float(xy[i, 0])
        y += float(shape[i]) * float(xy[i, 1])
    return x, y


def svg_text(x, y, s, size=14, anchor='middle', weight='normal'):
    s = str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def color(v, vmin, vmax):
    # blue -> near-white -> red diverging scale
    t = 0.5 if vmax == vmin else (v-vmin)/(vmax-vmin)
    if t < 0.5:
        q = t*2
        r, g, b = int(35+220*q), int(90+165*q), 220
    else:
        q = (t-0.5)*2
        r, g, b = 255, int(255-205*q), int(255-205*q)
    return f'#{r:02x}{g:02x}{b:02x}'


def save_shape_svg(xy, title, path):
    W, H = 760, 600
    pad = 90
    xmin, xmax = float(xy[:,0].min()), float(xy[:,0].max())
    ymin, ymax = float(xy[:,1].min()), float(xy[:,1].max())
    dx, dy = max(xmax-xmin, 1e-9), max(ymax-ymin, 1e-9)
    scale = min((W-2*pad)/dx, (H-2*pad)/dy)
    cx, cy = W/2, H/2
    def pt(x,y): return cx+(x-(xmin+xmax)/2)*scale, cy-(y-(ymin+ymax)/2)*scale
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="white"/>', svg_text(W/2, 38, title+' — element shape', 22, weight='bold')]
    if len(xy) == 4:
        boundary = np.vstack((xy,xy[0]))
    else:
        vals = np.linspace(-1,1,40)
        par = [(v,-1) for v in vals] + [(1,v) for v in vals] + [(v,1) for v in vals[::-1]] + [(-1,v) for v in vals[::-1]]
        boundary = np.array([map_to_physical(shape_functions(a,b,8), xy) for a,b in par])
    coords = [pt(*p) for p in boundary]
    points = ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
    parts.append(f'<polyline points="{points}" fill="#dce9f5" stroke="#174a70" stroke-width="3"/>')
    for i,p in enumerate(xy,1):
        x,y=pt(*p); parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#c43b32"/>')
        parts.append(svg_text(x+10,y-10,f'{i}',14,'start','bold'))
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def save_jacobian_svg(xg, yg, det, xphys, yphys, title, path):
    W,H=1200,610
    left=(65,105,500,430); right=(650,105,500,430)
    vmin,vmax=float(det.min()),float(det.max())
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',svg_text(W/2,35,title+' — determinant of Jacobian',22,weight='bold'),svg_text(315,78,'Parent domain (ξ, η)',18,weight='bold'),svg_text(900,78,'Physical domain (x, y)',18,weight='bold')]
    # Parent heatmap
    x0,y0,pw,ph=left
    nr,nc=det.shape
    for r in range(nr-1):
        for c in range(nc-1):
            xx=x0+c*pw/(nc-1); yy=y0+r*ph/(nr-1)
            ww=pw/(nc-1)+0.5; hh=ph/(nr-1)+0.5
            val=float((det[r,c]+det[r,c+1]+det[r+1,c]+det[r+1,c+1])/4.0)
            parts.append(f'<rect x="{xx:.2f}" y="{yy:.2f}" width="{ww:.2f}" height="{hh:.2f}" fill="{color(val,vmin,vmax)}"/>')
    parts += [f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',svg_text(x0+pw/2,y0+ph+28,'ξ',16),svg_text(x0-30,y0+ph/2,'η',16)]
    # Physical mapped heatmap using cell polygons
    x0,y0,pw,ph=right
    xmin,xmax=float(xphys.min()),float(xphys.max()); ymin,ymax=float(yphys.min()),float(yphys.max())
    dx=max(xmax-xmin,1e-9); dy=max(ymax-ymin,1e-9)
    scale=min(pw/dx,ph/dy)*0.94
    ox=x0+pw/2-(xmin+xmax)/2*scale; oy=y0+ph/2+(ymin+ymax)/2*scale
    def ptx(x,y): return ox+x*scale, oy-y*scale
    for r in range(nr-1):
        for c in range(nc-1):
            pts=[ptx(xphys[r,c],yphys[r,c]),ptx(xphys[r,c+1],yphys[r,c+1]),ptx(xphys[r+1,c+1],yphys[r+1,c+1]),ptx(xphys[r+1,c],yphys[r+1,c])]
            poly=' '.join(f'{a:.2f},{b:.2f}' for a,b in pts)
            val=float((det[r,c]+det[r,c+1]+det[r+1,c]+det[r+1,c+1])/4.0)
            parts.append(f'<polygon points="{poly}" fill="{color(val,vmin,vmax)}" stroke="none"/>')
    parts += [f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',svg_text(x0+pw/2,y0+ph+28,'x',16),svg_text(x0-30,y0+ph/2,'y',16)]
    # Color bar + range
    bx,by,bw,bh=555,130,24,370
    for k in range(100):
        yy=by+k*bh/100
        parts.append(f'<rect x="{bx}" y="{yy:.2f}" width="{bw}" height="{bh/100+1:.2f}" fill="{color(vmax-k*(vmax-vmin)/99,vmin,vmax)}"/>')
    parts += [svg_text(bx+bw+8,by+5,f'{vmax:.3f}',12,'start'),svg_text(bx+bw+8,by+bh,f'{vmin:.3f}',12,'start'),svg_text(W/2,585,f'det(J) range: {vmin:.4f} to {vmax:.4f}',16)]
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def calculate_and_save(xy, title, prefix):
    n=len(xy); points=np.linspace(-1,1,81)
    xi,eta=np.meshgrid(points,points)
    det=np.empty_like(xi); xp=np.empty_like(xi); yp=np.empty_like(xi)
    for r in range(len(points)):
        for c in range(len(points)):
            a,b=xi[r,c],eta[r,c]
            J=jacobian(a,b,xy)
            det[r,c]=J[0][0]*J[1][1]-J[0][1]*J[1][0]
            xp[r,c],yp[r,c]=map_to_physical(shape_functions(a,b,n), xy)
    print(f'{title}: det(J) min = {det.min():.4f}, max = {det.max():.4f}',flush=True)
    shape_path=OUT/f'{prefix}_fig1.svg'
    jac_path=OUT/f'{prefix}_fig2.svg'
    save_shape_svg(xy,title,shape_path)
    save_jacobian_svg(xi,eta,det,xp,yp,title,jac_path)
    print(f'Saved: {shape_path}',flush=True); print(f'Saved: {jac_path}',flush=True)
    return [shape_path,jac_path]


def main():
    print('Starting HW10Problem2',flush=True)
    all_paths=[]
    xy_q4=np.array([[2.,1.],[0.,0.],[1.,0.],[2.,-1.]])
    print('Calculating Q4 element (Question 2a)...',flush=True)
    all_paths += calculate_and_save(xy_q4,'Q4 element','a_q4')
    xy_q8=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,0.],[1.,.5],[.5,1.],[0.,.5]])
    print('Calculating baseline Q8 element...',flush=True)
    all_paths += calculate_and_save(xy_q8,'Q8 element','b0_q8_baseline')
    cases=[('b1_corner3_to_0.3_0.3',3,(.3,.3)),('b2_mid5_to_0.1_0',5,(.1,0.)),('b3_mid5_to_0.5_0.4_valid',5,(.5,.4)),('b4_mid6_to_1_0.9',6,(1.,.9))]
    for name,node,pos in cases:
        moved=xy_q8.copy(); moved[node-1]=pos
        print(f'Calculating {name}...',flush=True)
        all_paths += calculate_and_save(moved,f'Q8, node {node} -> {pos}',name)
    print(f'Done. All six cases calculated. {len(all_paths)} SVG plots saved in: {OUT}',flush=True)
    if os.name=='nt':
        print('Opening the first plot in your browser; all SVGs are saved in the same folder.',flush=True)
        try: os.startfile(str(all_paths[0]))
        except Exception as exc: print(f'Could not auto-open plot: {exc}',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        traceback.print_exc()
        input('Script failed. Press Enter to close...')
