## 1. Hardware System Prototype

To validate the in-situ characterization methodology proposed in the paper, a fully functional sensor prototype was developed and integrated into a self-developed 5-axis polishing robot. The figure below illustrates the physical implementation details, corresponding to the schematic design presented in **Fig. 1** of the manuscript.

![Hardware System Overview](./images/hardware_prototype.png)
*(Note: Please replace `./images/hardware_prototype.png` with the actual path to your image file in the repository)*

The hardware system consists of three key components:

### A. System Integration (Left)
*   **In-situ Setup:** The vision-based sensor is designed as a compact end-effector, mounted directly onto the spindle of the 5-axis polishing robot.
*   **Operation Context:** The figure demonstrates the sensor aligned with a **BK7 optical glass workpiece**. This setup enables the characterization of contact force and pose misalignment within the actual machining coordinate system, eliminating errors associated with workpiece remounting.

### B. Imaging Module (Top Right)
*   **Camera Unit:** An industrial CMOS camera (**OV2720**) is integrated into the base, featuring **1080P resolution** and **30 fps** capability. This high frame rate is leveraged for multi-frame averaging to suppress electronic noise.
*   **Illumination:** A custom **12-LED ring array** provides uniform, shadow-free internal lighting.
*   **Structural Design:** The aluminum alloy base incorporates a hermetic sealing structure, ensuring the module withstands the internal inflation pressure and rotational centrifugal loads.

### C. Sensing Bonnet (Bottom Right)
*   **Fabrication:** The flexible bonnet is manufactured via precision mold casting using silicone rubber (Shore 40A).
*   **Embedded Markers:** An array of **65 rigid markers** (2.0 mm diameter) is precisely embedded into the inner surface. This design ensures that the markers accurately represent the substrate deformation while being protected from the abrasive polishing interface.
