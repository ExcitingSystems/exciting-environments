Environments
############

On this page, all environments with their environment-id are listed.

=================================================================== ==============================
Environment                                                         environment-id
=================================================================== ==============================
**Basic Environments**

CartPole                                                             ``'CartPole-v0'``
MassSpringDamper                                                     ``'MassSpringDamper-v0'``
Pendulum                                                             ``'Pendulum-v0'``

=================================================================== ==============================

.. toctree::
   :maxdepth: 4
   :caption: Environments:
   :glob:

   pendulum/pendulum_env
   mass_spring_damper/mass_spring_damper_env
   cart_pole/cart_pole_env


Environment Structur
'''''''''''''''''''''''''''''''

.. autoclass:: exciting_environments.env_struct.CoreEnvironment
   :members: