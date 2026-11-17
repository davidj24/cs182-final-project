"""T01 repair 2: measure translation-only scene candidates on the 5 corner cases.
Run: python docs/evidence/T01_candidates.py   (seconds). Prints normal force, friction ratio, height."""
import mujoco, numpy as np

HEAD = '''<mujoco><compiler angle="degree"/>
<option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
<worldbody><geom name="floor" type="plane" size="0 0 0.1" friction="0 0 0" condim="3" solimp="0.99 0.999 0.001"/>
<body name="cube" pos="0 0 0.025">'''
TAIL = '''<inertial pos="0 0 0" mass="1" diaginertia="0.000416667 0.000416667 0.000416667"/>
<geom name="cube_geom" type="box" size="0.025 0.025 0.025" friction="0.4 0.005 0.0001" condim="3" solimp="0.99 0.999 0.001"/>
</body></worldbody></mujoco>'''
J = {
 "i_xy_only": '<joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>',
 "ii_xyz_free_z": '<joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><joint name="z" type="slide" axis="0 0 1"/>',
 "ii_xyz_zrange": '<joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><joint name="z" type="slide" axis="0 0 1" limited="true" range="-0.0005 0.0005"/>',
}
CORNERS = [(0.5,0.6,26,200),(0.5,0.3,26,200),(1.0,0.45,20,150),(1.5,0.6,26,200),(0.5,0.6,13,100)]

def run(xml, m, mu, F, k, nj):
    mdl = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(mdl)
    b = 1; mdl.body_mass[b] = m; mdl.body_inertia[b] = m*0.05**2/6; mdl.geom_friction[1,0] = mu
    mujoco.mj_setConst(mdl, d)
    for _ in range(500): mujoco.mj_step(mdl, d)
    z0 = d.qpos[2] if nj >= 3 else 0.025
    N=[];R=[];Z=[];nc=[]
    f6 = np.zeros(6)
    for i in range(k):
        d.xfrc_applied[b,0] = F; mujoco.mj_step(mdl, d)
        nc.append(d.ncon)
        if d.ncon:
            n = 0; t = 0
            for c in range(d.ncon):
                mujoco.mj_contactForce(mdl, d, c, f6); n += f6[0]; t += np.hypot(f6[1], f6[2])
            N.append(n); R.append(t/n if n > 1e-9 else np.nan)
        Z.append(d.qpos[2] if nj >= 3 else d.xpos[b][2])
    return dict(minnc=min(nc), N=(np.nanmin(N) if N else 0)/(m*9.81), Nmax=(np.nanmax(N) if N else 0)/(m*9.81),
                ratio=np.nanmedian(R) if R else np.nan, Rdev=np.nanmax(np.abs(np.array(R)/mu-1)) if R else np.nan,
                exc_mm=(max(Z)-min(Z))*1e3, zrest=Z[0])

if __name__ == "__main__":
    for name, j in J.items():
        xml = HEAD + j + TAIL; nj = j.count("<joint")
        print("==", name)
        for c in CORNERS:
            print(c, {kk: round(float(v), 5) for kk, v in run(xml, *c, nj).items()})
