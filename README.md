# A Novel Bonnet Polishing Approach with a Vision-Based Sensor for In-Situ Characterization of Contact Force and Pose Misalignment

Created by Feiyu Zhang, Jieji Ren, Langlang Yuan, Mengqi Rao, Yuehong Yin

## Introduction

This work is based on our CIRP 2026 paper titled _**A Novel Bonnet Polishing Approach with a Vision-Based Sensor for In-Situ Characterization of Contact Force and Pose Misalignment**_. We propose an embedded vision-based sensor utilizing inner-surface marker tracking to achieve in-situ perception of the contact state. This approach realizes precise pose error compensation and significant surface quality optimization.

## 1. Hardware System Prototype

To validate the in-situ characterization methodology proposed in the paper, a vision-based sensor was developed and integrated into a self-developed 5-axis robot arm.

<div align="center">
  <img src="./img/hardware_overview.png" alt="Overall Hardware System Architecture" width="800"/>
  <br />
  <b>Figure 1: Overall Hardware System Architecture.</b>
</div>

<div align="center">
  <img src="./img/hardware_prototype.png" alt="Hardware System Overview" width="800"/>
  <br />
  <b>Figure 2: Physical Implementation Details.</b>
</div>

## 2. Algorithm Verification

The proposed approach converts raw visual signals into quantitative contact state information via four sequential stages. Below, we detail the implementation and validation results for each stage.

### A. 2D Feature Extraction (Sub-pixel Capture)
The raw image captured by the internal camera is processed to extract marker centroids and IDs. The specific implementation is available in the [`Marker_Tracking`](./code/Marker_Tracking) directory. Figure 3 illustrates the dynamic 2D feature extraction process and the recognized 2D marker array from a static frame.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./img/2d_dynamic_capture.gif" alt="Dynamic 2D Feature Extraction" width="380"/>
        <br />
        <b>Figure 3 (a): Dynamic 2D Feature Extraction.</b>
      </td>
      <td align="center">
        <img src="./img/2d_visualization.png" alt="2D Recognition Result" width="400"/>
        <br />
        <b>Figure 3 (b): 2D Recognition (Static Frame).</b>
      </td>
    </tr>
  </table>
</div>

### B. 3D Displacement Field Reconstruction
Based on the 2D feature changes, the 3D displacement vectors are reconstructed using geometric constraints. The complete reconstruction pipeline is provided in the [`Marker_Calibration`](./code/Marker_Calibration) directory:

*   **Intrinsic Matrix Fitting:** [`intrinsic_calibration.py`](./code/Marker_Calibration/intrinsic_calibration.py) for camera parameter estimation (Figure 4 (a)).
*   **Extrinsic Matrix Fitting:** [`extrinsic_calibration.py`](./code/Marker_Calibration/extrinsic_calibration.py) for coordinate system alignment (Figure 4 (b)).
*   **3D Reconstruction:** [`3d_reconstruction.py`](./code/Marker_Calibration/3d_reconstruction.py) for resolving the spatial displacement field (Figure 4 (c)).

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./img/intrinsic.png" alt="Intrinsic Calibration" width="325"/>
        <br />
        <b>Figure 4 (a): Intrinsic Calibration.</b>
      </td>
      <td align="center">
        <img src="./img/extrinsic.png" alt="Extrinsic Calibration" width="490"/>
        <br />
        <b>Figure 4 (b): Extrinsic Calibration.</b>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <img src="./img/3d_vector_placeholder.png" alt="3D Displacement Vector Field" width="700"/>
  <br />
  <b>Figure 4 (c): 3D Displacement Vector Field.</b>
</div>


### C. Precision Validation
To ensure measurement reliability, we performed the following two-step accuracy validation:

#### 1. Geometric Validation (Marker Diameter)
First, the absolute diameter of the rigid markers was validated (Figure 5).

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./img/diameter_shot.png" alt="Marker Shot" width="240"/>
        <br />
        <b>Figure 5 (a): Markers Photo.</b>
      </td>
      <td align="center">
        <img src="./img/diameter_histogram.png" alt="Accuracy Histogram" width="590"/>
        <br />
        <b>Figure 5 (b): Diameter Estimation Accuracy.</b>
      </td>
    </tr>
  </table>
</div>

#### 2. Mechanical Validation (Probe Indentation)
To validate the algorithmic accuracy, a precision probe indentation test was performed. The tool was pressed in **steps of 0.7 mm (12 steps)**. Figure 6 compares the prescribed probe depth with the algorithm-calculated displacement.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./img/Experimental_Platform.png" alt="Probe Indentation Setup" width="240"/>
        <br />
        <b>Figure 6 (a): Experimental Platform.</b>
      </td>
      <td align="center">
        <img src="./img/Sensor_Error_Analysis.png" alt="Stepwise Accuracy" width="540"/>
        <br />
        <b>Figure 6 (b): Sensor Error Analysis.</b>
      </td>
    </tr>
  </table>
</div>

### D. Pose Misalignment Identification
This module implements the reference state comparison strategy. By analyzing the deviation of the 3D displacement field, a spatial contact plane is fitted to identify the tilt angle.

<div align="center">
  <img src="./img/MarkerDistribution1.png" alt="Pose Fitting Result" width="700"/>
  <br />
  <b>Figure 7: Pose Fitting Result.</b>
</div>


## 3. Vision-based Sensor Operation Demos

This section demonstrates the vision-based sensor's response under different contact conditions.

### A. Vertical Compression
In this scenario, the bonnet tool is pressed vertically against the workpiece ($\psi = 0^\circ$). Figure 8 illustrates the outside view and the internal camera view during vertical compression.

<div align="center">
  <img src="./img/vertical_compression.gif" width="800"/>
  <div><b>Figure 8: Vertical Compression Demo.</b></div>
</div>

### B. Tilted Compression
In this scenario, the tool is pressed against the workpiece surface at a standard polishing precession angle of **15°**. The sensor captures the marker displacement characteristic of this pose, which serves as the reference state for subsequent misalignment detection. Figure 9 illustrates the physical setup and the internal camera view during tilted compression.

<div align="center">
  <img src="./img/tilted_compression.gif" width="800"/>
  <div><b>Figure 9: Tilted Compression Demo.</b></div>
</div>

### C. Dynamic Polishing Process
This demo shows the sensor operation during the polishing process. The high-speed camera captures stable marker features despite the rotation, validating the robustness of the imaging module. Figure 10 illustrates the polishing process and the dynamic internal camera view.

<div align="center">
  <img src="./img/dynamic_polishing.gif" width="800"/>
  <div><b>Figure 10: Dynamic Polishing Process Demo.</b></div>
</div>

The following figures further demonstrate the stability of the displacement extracted by our algorithms during the dynamic polishing process.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./img/total_marker_displacement.png" alt="Total Marker Displacement (Filtered)" width="307"/>
        <br />
        <b>Figure 11 (a): Total Marker Displacement.</b>
      </td>
      <td align="center">
        <img src="./img/point_polishing.png" alt="Amplitude Displacement of Individual Markers" width="493"/>
        <br />
        <b>Figure 11 (b): Amplitude Displacement of Individual Markers.</b>
      </td>
    </tr>
  </table>
</div>