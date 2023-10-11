Cartpole
==================================================

Description
--------------------------------------------------
Environment to simulate a Cartpole System.

Equation
--------------------------------------------------
.. math::

      u= l m g sin(\theta) + l^2 m \ddot{\theta}

Parameters
--------------------------------------------------
   
| :math:`\mu_{p}`: Coefficient of friction of pole on cart
| :math:`\mu_{c}`: Coefficient of friction of cart on track
| :math:`l`: Half-pole length
| :math:`m_{c}`: Mass of cart
| :math:`m_{p}`: Mass of pole
| :math:`g`: Gravitational acceleration

States
--------------------------------------------------

====== ========================  ==============
Num    Term in Equation          Term in Class            
====== ========================  ==============
0      :math:`x`                 deflection
1      :math:`\dot{x}`           velocity
2      :math:`\theta`            theta
3      :math:`\dot{\theta}`      omega
====== ========================  ==============

Class
--------------------------------------------------

.. autoclass:: exciting_environments.cart_pole.cart_pole_env.CartPole
   :members: