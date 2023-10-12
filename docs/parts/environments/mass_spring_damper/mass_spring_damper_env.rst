Mass Spring Damper
==================================================

Description
--------------------------------------------------
Environment to simulate a Mass-Spring-Damper System.

Equation
--------------------------------------------------
.. math::

      u = m \ddot{y} + d \dot{y} + k y

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
0      :math:`y`                 deflection
1      :math:`\dot{y}`           velocity
====== ========================  ==============

Class
--------------------------------------------------

.. autoclass:: exciting_environments.mass_spring_damper.mass_spring_damper_env.MassSpringDamper
   :members:
