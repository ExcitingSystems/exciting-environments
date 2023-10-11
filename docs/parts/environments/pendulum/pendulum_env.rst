Pendulum
==================================================

Description
--------------------------------------------------
Environment to simulate a simple Pendulum.

Equation
--------------------------------------------------
.. math::

      u= l m g sin(\theta) + l^2 m \ddot{\theta}

Parameters
--------------------------------------------------
   
| :math:`l`:Length of the pendulum 
| :math:`m`:Mass of the pendulum tip 
| :math:`g`: Gravitational acceleration

State-/ Obvservation Space
--------------------------------------------------

====== ==========================  ==============
Num    Name in Equation            Term in Class            
====== ==========================  ==============
0      :math:`\theta`              theta
1      :math:`\ddot{\theta}`       omega
====== ==========================  ==============

Class
--------------------------------------------------

.. autoclass:: exciting_environments.pendulum.pendulum_env.Pendulum
   :members: