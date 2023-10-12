Mass Spring Damper
==================================================

Description
--------------------------------------------------
Environment to simulate a Mass-Spring-Damper System.

Equation
--------------------------------------------------
.. math::

      u_t = m \ddot{y_t} + d \dot{y_t} + k y_t

Parameters
--------------------------------------------------
   
| :math:`d`: Damping constant
| :math:`k`: Spring constant
| :math:`m`: Mass of the oscillating object

States
--------------------------------------------------

====== ========================  ==============
Num    Term in Equation          Term in Class            
====== ========================  ==============
0      :math:`y_t`                 deflection
1      :math:`\dot{y_t}`           velocity
====== ========================  ==============

Class
--------------------------------------------------

.. autoclass:: exciting_environments.mass_spring_damper.mass_spring_damper_env.MassSpringDamper
   :members:
