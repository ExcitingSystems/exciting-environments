Pendulum
==================================================

Description
--------------------------------------------------
Environment to simulate a simple Pendulum.

Equation
--------------------------------------------------
.. math::

      M_t = l m g \sin{\theta_t} + l^2 m \ddot{\theta_t}

Parameters
--------------------------------------------------
   
| :math:`l`: Length of the pendulum 
| :math:`m`: Mass of the pendulum tip 
| :math:`g`: Gravitational acceleration

Action
--------------------------------------------------

====== ========================  ==============
Num    Term in Equation          Term in Class            
====== ========================  ==============
0      :math:`M_t`                 torque
====== ========================  ==============


States
--------------------------------------------------

====== ========================  ==============
Num    Term in Equation          Term in Class            
====== ========================  ==============
0      :math:`\theta_t`            theta
1      :math:`\dot{\theta_t}`      omega
====== ========================  ==============

Class
--------------------------------------------------

.. autoclass:: exciting_environments.pendulum.pendulum_env.Pendulum
   :members: