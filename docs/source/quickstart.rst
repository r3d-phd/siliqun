Quick Start
===========

This guide demonstrates the core SiliQun workflow: creating an environment, training a DRL agent, and evaluating the results.

Creating an Environment
-----------------------

.. code-block:: python

   import siliqun

   # 3-qubit GHZ state preparation with SiMOS device profile
   env = siliqun.SiliQunEnv(
       n_qubits=3,
       device="SiMOS",
       target="GHZ",
       backend="auto",   # auto-selects MPS or GPU state vector
       max_steps=20,
   )

   obs, info = env.reset(seed=42)
   print(f"Observation shape: {obs.shape}")   # (n_obs_dim,)
   print(f"Action space: {env.action_space}") # Box(-1, 1, (n_actions,))

Running a Random Policy
-----------------------

.. code-block:: python

   obs, info = env.reset()
   total_reward = 0

   for step in range(20):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       total_reward += reward
       if terminated or truncated:
           break

   print(f"Total reward: {total_reward:.4f}")
   print(f"Final fidelity: {info['fidelity']:.4f}")

Training with Stable-Baselines3
---------------------------------

.. code-block:: python

   from stable_baselines3 import SAC
   import siliqun

   env = siliqun.SiliQunEnv(n_qubits=3, device="SiMOS", target="GHZ")

   model = SAC(
       "MlpPolicy",
       env,
       learning_rate=3e-4,
       buffer_size=100_000,
       batch_size=256,
       verbose=1,
   )
   model.learn(total_timesteps=500_000)

   # Evaluate
   obs, _ = env.reset()
   for _ in range(20):
       action, _ = model.predict(obs, deterministic=True)
       obs, reward, done, _, info = env.step(action)
       if done:
           break
   print(f"Evaluation fidelity: {info['fidelity']:.4f}")

Available Device Profiles
--------------------------

.. code-block:: python

   # Donor spin qubit (P-in-Si)
   env = siliqun.SiliQunEnv(n_qubits=2, device="Donor")

   # Silicon MOS qubit (Intel/UNSW style)
   env = siliqun.SiliQunEnv(n_qubits=3, device="SiMOS")

   # Gate-all-around nanowire qubit
   env = siliqun.SiliQunEnv(n_qubits=4, device="GAA")

   # RIKEN 5-qubit device (Noiri et al. 2022)
   env = siliqun.SiliQunEnv(n_qubits=5, device="RIKEN5Q")

See :doc:`device_profiles` for full parameter tables.

SLEDGE Grid Topologies
-----------------------

.. code-block:: python

   # 2x2 grid (4 physical qubits → 2 logical DFS qubits)
   env = siliqun.SiliQunEnv(n_qubits=4, topology="2x2", use_dfs=True)

   # 3x3 grid (9 physical qubits → 4 logical DFS qubits)
   env = siliqun.SiliQunEnv(n_qubits=9, topology="3x3", use_dfs=True)
