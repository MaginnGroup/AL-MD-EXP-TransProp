
variable         NAME index hc
log              ${NAME}.log
units            real
atom_style       full
pair_style	 lj/cut 14.0

bond_style	 harmonic
angle_style	 harmonic
dihedral_style	 opls

read_data        box.lmp

kspace_style    none

pair_modify     mix geometric tail yes

dump            1 all custom 100000 ${NAME}.lammpstrj id type xu yu zu

thermo_style    custom step vol temp press ke pe evdwl lx ly lz density
thermo          200

timestep        1.0

velocity        all create 303.0 123456

minimize        1.0e-4 1.0e-6 1000 1000

fix             npt1 all npt temp 303.0 303.0 100.0 iso 98.6923 98.6923 1000.0

run             2000000

reset_timestep  0

variable        lx equal lx
fix             3 all ave/time 200 25000 5000000 v_lx ave running
variable        mean_lx equal f_3

run             5000000

unfix           npt1

change_box      all x final 0.0 ${mean_lx} y final 0.0 ${mean_lx} z final 0.0 ${mean_lx} remap units box

fix             nvta all nvt temp 303.0 303.0 100.0

run             2000000

write_data      data.${NAME}

write_restart   restart.${NAME}

