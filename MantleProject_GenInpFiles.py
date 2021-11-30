def generate_input( expansion_coeff, nu0, ref_temp, beta, fname):
  input_string = """# A simple setup for convection in a quarter of a 2d shell. See the
  # manual for more information.


  set Dimension                              = 2
  set Use years in output instead of seconds = true
  set End time                               = 0.5e9
  set Output directory                       = ./output_{4} #./output-shell_simple_2d_edits_T0_new9


  subsection Material model
    set Model name = simple

    subsection Simple model
      set Thermal expansion coefficient = {0} # 4e-5
      set Viscosity                     = {1} # 5e21  #3.3e21
      set Reference temperature         = {2} # 1500 #1600
      set Thermal viscosity exponent    = {3} # 0.7

    end
  end


  subsection Geometry model
    set Model name = spherical shell

    subsection Spherical shell
      set Inner radius  = 3473000
      set Outer radius  = 6301000
      set Opening angle = 90
    end
  end


  subsection Boundary velocity model
    set Zero velocity boundary indicators       = inner
    set Tangential velocity boundary indicators = top, left, right
  end


  subsection Heating model
    set List of model names =  shear heating
  end


  subsection Boundary temperature model
    set Fixed temperature boundary indicators = top, bottom
    set List of model names = spherical constant

    subsection Spherical constant
      set Inner temperature = 3974 #3700
      set Outer temperature = 774 #500
    end
  end


  subsection Initial temperature model
    set Model name = function

    subsection Function
      set Variable names = x,y
      set Function constants = k=1e-6, \
                              tau=3e16, \
                              r_in = 3473000, \
                              T_1=774, T_0=3974, \
                              r_outer=6301000
      set Function expression = T_1 + (T_0-T_1)*(1-erfc((r_outer - sqrt(x*x + y*y)) /sqrt(k*tau))) + (rand()/32767-0.5)*(3200*0.01)
    end
  end


  subsection Gravity model
    set Model name = ascii data
  end


  subsection Mesh refinement
    set Initial global refinement          = 4
    set Strategy                           = temperature
    set Time steps between mesh refinement = 0
  end


  subsection Postprocess
    set List of postprocessors = visualization, velocity statistics, temperature statistics, heat flux statistics, depth average

    subsection Visualization
      set Output format                 = vtu
      set Time between graphical output = 25e6
      set Number of grouped files       = 0
      set List of output variables      = material properties
    end

    subsection Depth average
      set Time between graphical output = 25e6
    end
  end""".format( expansion_coeff, nu0, ref_temp, beta, fname )

  # print(input_string)

  with open(fname+'.prm', 'w') as f:
    f.write(input_string)

# counter = 0
# for alpha in ['4.3e-5', '3.61e-5', '2.47e-5', '2.01e-5', '1.55e-5']:
#   for nu in ['1e21', '2e21', '3e21', '4e21', '5e21']:
#     for ref_T in ['1300', '1400', '1450', '1500', '1600']:
#         for visc_exp_beta in ['0.0', '0.2', '0.4', '0.6', '0.8']:
#             fname = "mantle_config_{0:04d}".format(counter)
#             generate_input( alpha, nu, ref_T, visc_exp_beta, fname )
#             counter += 1


# additional inputs
counter = 625
for alpha in ['4.3e-5', '3.61e-5', '2.47e-5', '2.01e-5', '1.55e-5']:
  for nu in ['1e21', '2e21', '3e21', '4e21', '5e21']:
    for ref_T in ['1300', '1400', '1450', '1500', '1600']:
        for visc_exp_beta in ['0.1', '0.3', '0.5', '0.7']:
            fname = "mantle_config_{0:04d}".format(counter)
            generate_input( alpha, nu, ref_T, visc_exp_beta, fname )
            counter += 1