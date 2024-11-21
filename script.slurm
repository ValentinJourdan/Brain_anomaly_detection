#!/bin/bash
#SBATCH --job-name=test_job           # Nom du job
#SBATCH --output=output_%j.txt        # Fichier de sortie (%j = ID du job)
#SBATCH --error=error_%j.txt          # Fichier d'erreurs
#SBATCH --partition=gpu_inter         # Partition (par ex. cpu, gpu)
#SBATCH --ntasks=1                    # Nombre total de tâches
#SBATCH --cpus-per-task=4             # Nombre de CPUs par tâche
#SBATCH --mem=16G                     # Mémoire allouée
#SBATCH --time=01:00:00               # Temps maximum d'exécution (HH:MM:SS)

# Chargez les modules nécessaires
module load python/3.8 cuda/11.3

# Exécutez votre programme
python3 Brain_anomaly_detection.py