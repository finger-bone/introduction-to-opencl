use ocl::ProQue;
use plotly::{common::{ColorScale, ColorScalePalette}, HeatMap, Layout, Plot};
use indicatif::ProgressBar;

const KERNEL_SRC: &str = include_str!("kernel.cl");

fn main() -> ocl::Result<()> {
    // Simulation parameters
    let size_x = 512;       // Grid size in x-direction
    let size_y = 512;       // Grid size in y-direction
    let dt = 0.001f32;          // Time step
    let dl = 1f32;           // Spatial step
    let a0 = 10.0f32;           // α₀ coefficient
    let a1 = 0.01f32;         // α₁ coefficient
    let a2 = 0.0001f32;        // α₂ coefficient
    let num_steps = 1000000;    // Number of time steps

    // Initialize OpenCL
    let pro_que = ProQue::builder()
        .src(KERNEL_SRC)
        .dims((size_x, size_y))
        .build()?;

    // Create initial temperature field
    let temp_data = vec![0.0f32; size_x * size_y];

    // Create double buffers
    let buffers = vec![
        pro_que.create_buffer::<f32>()?,
        pro_que.create_buffer::<f32>()?,
    ];

    // Initialize buffers
    buffers[0].write(&temp_data).enq()?;
    buffers[1].write(&temp_data).enq()?;
    // Create kernel
    let kernel = pro_que.kernel_builder("next_step")
        .arg(&buffers[0])  // Initial T_current buffer
        .arg(&buffers[1])  // Initial T_next buffer
        .arg(dt)
        .arg(dl)
        .arg(a0)
        .arg(a1)
        .arg(a2) 
        .local_work_size([16, 16])
        .build()?;

    let pb = ProgressBar::new(num_steps as u64);
    let mut computation_time = std::time::Duration::new(0, 0);
    let mut start_time = std::time::Instant::now();
    // Simulation loop
    for step in 0..num_steps {
        pb.inc(1);
        let (current_idx, next_idx) = (step % 2, (step + 1) % 2);
        
        // Set kernel arguments
        kernel.set_arg(0, &buffers[current_idx])?;
        kernel.set_arg(1, &buffers[next_idx])?;
        // Execute kernel
        unsafe { kernel.enq()?; }

        // Read results
        let mut temp_result = vec![0.0f32; size_x * size_y];
        buffers[next_idx].read(&mut temp_result).enq()?;
        computation_time += start_time.elapsed();
        start_time = std::time::Instant::now();
        // Visualize
        if step % (num_steps / 10) == ((num_steps / 10) - 1){
            plot_temperature(&temp_result, size_x, size_y, step).unwrap();
        }
    }
    println!("Time elapsed: {:.2?}", computation_time);

    Ok(())
}

fn plot_temperature(
    temperature: &[f32],
    size_x: usize,
    size_y: usize,
    step: usize
) -> Result<(), Box<dyn std::error::Error>> {
    // Prepare data matrix (transposed for correct orientation)
    let z: Vec<Vec<f32>> = (0..size_y)
        .map(|y| (0..size_x).map(|x| temperature[x * size_y + y]).collect())
        .collect();

    // Create heatmap trace
    let trace = HeatMap::new(
        (0..size_x).map(|x| x as i32).collect(),  // X coordinates
        (0..size_y).map(|y| y as i32).collect(),  // Y coordinates
        z
    )
    .name("Temperature")
    .color_scale(ColorScale::Palette(ColorScalePalette::RdBu));

    // Create plot layout
    let layout = Layout::new()
        .title(format!("Heat Distribution - Step {}", step).as_str())
        .x_axis(plotly::layout::Axis::new().title("X Position"))
        .y_axis(plotly::layout::Axis::new().title("Y Position"))
        .width(800)
        .height(800);

    // Create and save plot
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.write_html(format!("output/heatmap_step_{}.html", step));

    Ok(())
}