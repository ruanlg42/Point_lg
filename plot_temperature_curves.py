#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for plotting original and reconstructed temperature curves
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt plugin issues
import matplotlib.pyplot as plt
import sys
import os

def plot_temperature_curves(txt_file):
    """
    Read txt file and plot three figures together:
    1. Original temperature curve (red)
    2. Reconstructed temperature curve (green)
    3. Comparison of both curves
    """
    
    # Check if file exists
    if not os.path.exists(txt_file):
        print(f"Error: File {txt_file} does not exist!")
        return
    
    # Read data
    data = np.loadtxt(txt_file, skiprows=1)  # Skip header line
    
    time = data[:, 0]
    original_temp = data[:, 1]
    reconstructed_temp = data[:, 2]
    
    # Calculate statistics
    mean_error = np.mean(np.abs(original_temp - reconstructed_temp))
    max_error = np.max(np.abs(original_temp - reconstructed_temp))
    rmse = np.sqrt(np.mean((original_temp - reconstructed_temp)**2))
    
    # Get output directory
    output_dir = os.path.dirname(txt_file)
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # Subplot 1: Original temperature curve (red)
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(time, original_temp, 'r-', linewidth=2, label='Original Temperature')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Temperature (K)', fontsize=12)
    ax1.set_title('Original Temperature', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Subplot 2: Reconstructed temperature curve (green)
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(time, reconstructed_temp, 'g-', linewidth=2, label='Reconstructed Temperature')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Temperature (K)', fontsize=12)
    ax2.set_title('Reconstructed Temperature', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Subplot 3: Comparison of both curves
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(time, original_temp, 'r-', linewidth=2, label='Original', alpha=0.8)
    ax3.plot(time, reconstructed_temp, 'g-', linewidth=2, label='Reconstructed', alpha=0.8)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Temperature (K)', fontsize=12)
    ax3.set_title(f'Comparison\nMAE: {mean_error:.4f} K | Max Error: {max_error:.4f} K', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{base_name}_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Time range: {time[0]:.2f} - {time[-1]:.2f} s")
    print(f"  Original temperature range: {np.min(original_temp):.4f} - {np.max(original_temp):.4f} K")
    print(f"  Reconstructed temperature range: {np.min(reconstructed_temp):.4f} - {np.max(reconstructed_temp):.4f} K")
    print(f"  Mean Absolute Error (MAE): {mean_error:.4f} K")
    print(f"  Max Absolute Error: {max_error:.4f} K")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f} K")

def process_directory(directory, plot_number=-1):
    """
    Process txt files in the specified directory
    
    Args:
        directory: Directory path containing txt files
        plot_number: Number of plots to generate (-1 for all files)
    """
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist!")
        return
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory!")
        return
    
    # Find all txt files in the directory
    txt_files = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])
    
    if not txt_files:
        print(f"No txt files found in {directory}")
        return
    
    # Determine how many files to process
    total_files = len(txt_files)
    if plot_number == -1:
        files_to_process = txt_files
        print(f"Found {total_files} txt file(s) in {directory}")
        print(f"Processing ALL files")
    else:
        files_to_process = txt_files[:plot_number]
        print(f"Found {total_files} txt file(s) in {directory}")
        print(f"Processing first {len(files_to_process)} file(s)")
    
    print("=" * 80)
    
    # Process each txt file
    success_count = 0
    failed_count = 0
    
    for i, txt_file in enumerate(files_to_process, 1):
        txt_path = os.path.join(directory, txt_file)
        print(f"\n[{i}/{len(files_to_process)}] Processing: {txt_file}")
        print("-" * 80)
        
        try:
            plot_temperature_curves(txt_path)
            success_count += 1
        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
            failed_count += 1
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total processed: {len(files_to_process)}")
    if plot_number != -1 and total_files > plot_number:
        print(f"  Skipped: {total_files - len(files_to_process)}")

if __name__ == "__main__":
    # Default settings
    plot_number = -1  # -1 means plot all files
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        # Optional: plot_number as second argument
        if len(sys.argv) > 2:
            try:
                plot_number = int(sys.argv[2])
            except ValueError:
                print(f"Warning: Invalid plot_number '{sys.argv[2]}', using -1 (all files)")
                plot_number = -1
        
        # Check if input is a directory or a file
        if os.path.isdir(input_path):
            process_directory(input_path, plot_number)
        elif os.path.isfile(input_path):
            plot_temperature_curves(input_path)
        else:
            print(f"Error: {input_path} does not exist!")
    else:
        # Default: process sample directory
        default_dir = "/home/ziwu/Newpython/lg_exp/Point_lg/results/test_result/AAA_GFilter_new_residual_dconv16_20251105_165202_timevae_hybrid_res_d128_latent32_mamba2_trans2_head8_ic0.5/samples"
        print(f"No input specified, using default directory: {default_dir}")
        print(f"plot_number: {plot_number} (change this in the script if needed)")
        process_directory(default_dir, plot_number)

