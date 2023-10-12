Cartpole
==================================================

Description
--------------------------------------------------
Environment to simulate a Cartpole System.

Equation
--------------------------------------------------
.. math::

      \ddot{\theta_t}= \frac{g\sin{\theta_t}+\cos{\theta_t}[\frac{-F_t-ml{\dot{\theta_t}}^2 \sin{\theta_t}+\mu_c \textrm{sgn}({\dot{x}})}{m_c+m_p}]-\frac{\mu_p \dot{\theta_t}}{m_p l}}{l [\frac{4}{3}- \frac{m_p {\cos^2{\theta_t}}}{m_c + m_p}]} \\
      \ddot{x_t}= \frac{u_t + m l [{\dot{\theta_t}}^2 \sin{\theta_t}- \ddot{\theta_t} \cos{\theta_t}]-\mu_c \textrm{sgn}(\dot{x_t})}{m_c + m_p}

Parameters
--------------------------------------------------
   
| :math:`\mu_{p}`: Coefficient of friction of pole on cart
| :math:`\mu_{c}`: Coefficient of friction of cart on track
| :math:`l`: Half-pole length
| :math:`m_{c}`: Mass of cart
| :math:`m_{p}`: Mass of pole
| :math:`g`: Gravitational acceleration

Action
--------------------------------------------------

====== ========================  ==============
Num    Term in Equation          Term in Class            
====== ========================  ==============
0      :math:`F_t`                 force
====== ========================  ==============


States
--------------------------------------------------

====== ========================  ==============
Num    Term in Equation          Term in Class            
====== ========================  ==============
0      :math:`x_t`                 deflection
1      :math:`\dot{x_t}`           velocity
2      :math:`\theta_t`            theta
3      :math:`\dot{\theta_t}`      omega
====== ========================  ==============

Class
--------------------------------------------------

.. autoclass:: exciting_environments.cart_pole.cart_pole_env.CartPole
   :members: