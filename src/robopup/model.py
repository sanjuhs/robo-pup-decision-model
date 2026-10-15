"""Original primitive quadruped. SI units, +X forward and +Z up."""


def arena_xml(obstacles: bool = False) -> str:
    legs, actuators = [], []
    for name, x, y in (("fl", .23, .145), ("fr", .23, -.145),
                       ("rl", -.23, .145), ("rr", -.23, -.145)):
        legs.append(f'''
        <body name="{name}_hip" pos="{x} {y} 0">
          <joint name="{name}_roll" axis="1 0 0" range="-.6 .6"/>
          <geom type="sphere" size=".045" mass=".12" rgba=".16 .21 .25 1"/>
          <body name="{name}_thigh">
            <joint name="{name}_pitch" axis="0 1 0" range="-1.2 1.4"/>
            <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".026" mass=".24"/>
            <body name="{name}_shin" pos="0 0 -.2">
              <joint name="{name}_knee" axis="0 1 0" range="-2.6 -.05"/>
              <geom type="capsule" fromto="0 0 0 0 0 -.21" size=".021" mass=".18" rgba=".18 .28 .3 1"/>
              <geom name="{name}_foot" type="sphere" pos="0 0 -.21" size=".034" mass=".045" friction="1 .01 .001" rgba=".07 .1 .12 1"/>
            </body>
          </body>
        </body>''')
        for joint in ("roll", "pitch", "knee"):
            actuators.append(f'<position joint="{name}_{joint}" kp="45" kv="2.5" forcerange="-18 18"/>')
    props = '''<body name="crate" pos="1.5 0 .14">
       <freejoint name="crate_free"/>
       <geom name="crate_geom" type="box" size=".14 .14 .14" mass=".35" friction=".45 .01 .001" rgba="1 .6 .2 1"/>
     </body>'''
    if obstacles:
        props += '''<geom name="obstacle_a" type="box" pos=".3 1.3 .2" size=".5 .16 .2" rgba=".45 .5 .65 1"/>
        <geom name="obstacle_b" type="cylinder" pos="-1.4 -.8 .2" size=".22 .2" rgba=".45 .5 .65 1"/>
        <body name="ball" pos="1.8 -1 .125"><freejoint name="ball_free"/>
          <geom name="ball_geom" type="sphere" size=".12" mass=".15" friction=".7 .01 .002" rgba=".96 .44 .34 1"/>
        </body>'''
    return f'''<mujoco model="RoboPup">
      <compiler angle="radian"/>
      <option timestep=".005" integrator="implicitfast" gravity="0 0 -9.81"/>
      <default>
        <joint damping=".15" armature=".015"/>
        <geom condim="3" friction=".9 .01 .001" rgba=".15 .78 .66 1"/>
      </default>
      <worldbody>
        <light pos="0 0 4"/>
        <geom name="floor" type="plane" size="12 12 .1" rgba=".12 .16 .2 1"/>
        <body name="torso" pos="0 0 .415">
          <freejoint name="root"/>
          <geom name="torso_geom" type="box" size=".26 .13 .075" mass="3"/>
          <geom name="head" type="box" pos=".31 0 .045" size=".09 .1 .075" mass=".25" rgba=".88 .95 .88 1"/>
          {''.join(legs)}
        </body>
        {props}
      </worldbody>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>'''
