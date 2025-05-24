use clap::Parser;
use image::{
    ExtendedColorType, ImageBuffer, ImageEncoder, Rgb,
    codecs::{avif::AvifEncoder, jpeg::JpegEncoder},
};
use image_compare::Similarity;
use rayon::prelude::*;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;
use webp::Encoder;

/// Image optimizer, similar to jpeg-recompress but in Rust!
#[derive(Parser)]
struct Arguments {
    /// Input images, it can be a single file or a directory of files.
    input: PathBuf,

    /// Output directory, where optimized images will be saved.
    output_dir: PathBuf,

    /// Expected quality in comparison to the original image, in the form of a SSIM score.
    #[arg(short, long, default_value_t = 0.995)]
    ssim: f64,

    /// Desired output image format, can be jpeg, jpg, avif and webp
    #[arg(short, long, default_value_t = String::from("jpg"))]
    image_format: String,
}

fn load_image(
    path: &Path,
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error + Send + Sync>> {
    let loaded_image = image::ImageReader::open(path)?.decode()?.to_rgb8();
    Ok(loaded_image)
}

fn calculate_ssim(
    img1: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    img2: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> Result<Similarity, Box<dyn std::error::Error + Send + Sync>> {
    let ssim_result = image_compare::rgb_hybrid_compare(img1, img2)?;
    Ok(ssim_result)
}

fn save_image(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    quality: u8,
    output_path: &Path,
    save_format: &str,
    speed: u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match save_format.to_ascii_lowercase().as_str() {
        "webp" => {
            let encoder = Encoder::from_rgb(img, img.width(), img.height());
            let webp_mem = encoder.encode(quality.into());
            std::fs::write(output_path, &*webp_mem)?;
        }
        "jpeg" | "jpg" => {
            let mut bytes: Vec<u8> = vec![];
            let writer = Cursor::new(&mut bytes);
            let mut encoder = JpegEncoder::new_with_quality(writer, quality);
            encoder.encode(img, img.width(), img.height(), ExtendedColorType::Rgb8)?;
            std::fs::write(output_path, &bytes)?;
        }
        "avif" => {
            let mut bytes: Vec<u8> = vec![];
            let writer = Cursor::new(&mut bytes);
            let encoder = AvifEncoder::new_with_speed_quality(writer, speed, quality);
            encoder.write_image(img, img.width(), img.height(), ExtendedColorType::Rgb8)?;
            std::fs::write(output_path, &bytes)?;
        }
        _ => return Err("Unsupported image format".into()),
    }
    Ok(())
}

fn get_ssim_for_chosen_quality(
    original_image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    quality: u8,
    temp_output_path: &Path,
    save_format: &str,
) -> Result<Similarity, Box<dyn std::error::Error + Send + Sync>> {
    save_image(original_image, quality, temp_output_path, save_format, 10)?;

    let temp_image = load_image(temp_output_path)?;

    let ssim_result = calculate_ssim(original_image, &temp_image)?;

    Ok(ssim_result)
}

fn search_optimal_quality(
    original_img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    target_ssim: f64,
    high: u8,
    low: u8,
    path: &Path,
    save_format: &str,
) -> Result<u8, Box<dyn std::error::Error + Send + Sync>> {
    let mid = (high + low) / 2;
    let mid_ssim = get_ssim_for_chosen_quality(original_img, mid, path, save_format)?.score;

    let error_margin = 0.0009;
    if (mid_ssim - target_ssim).abs() < error_margin || mid >= high || mid <= low {
        let result = mid.clamp(1, 100);
        Ok(result)
    } else if mid_ssim < target_ssim {
        search_optimal_quality(original_img, target_ssim, high, mid, path, save_format)
    } else {
        search_optimal_quality(original_img, target_ssim, mid, low, path, save_format)
    }
}

fn find_optimal_image_quality(
    original_img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    target_ssim: f64,
    save_format: &str,
) -> Result<u8, Box<dyn std::error::Error + Send + Sync>> {
    let temp_output_file = NamedTempFile::with_suffix(format!(".{save_format}"))?;
    let temp_output_path = temp_output_file.path();

    search_optimal_quality(
        original_img,
        target_ssim,
        100,
        1,
        temp_output_path,
        save_format,
    )
}

fn process_single_image(
    input_path: &Path,
    target_ssim: f64,
    output_path: &Path,
    save_format: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let image_filename = input_path.file_stem();
    if let Some(img_filename) = image_filename {
        let original_image = load_image(input_path)?;
        let quality_result = find_optimal_image_quality(&original_image, target_ssim, save_format)?;
        save_image(
            &original_image,
            quality_result,
            &output_path.join(format!(
                "{}.{}",
                img_filename.to_string_lossy(),
                save_format
            )),
            save_format,
            1,
        )?;
    }
    Ok(())
}

fn process_images(
    input_paths: &[PathBuf],
    target_ssim: f64,
    output_path: &Path,
    save_format: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    input_paths
        .par_iter()
        .map(
            |path_buf| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                process_single_image(path_buf, target_ssim, output_path, save_format)
            },
        )
        .collect()
}

fn main() {
    let args = Arguments::parse();

    if args.ssim < 0.0 || args.ssim > 1.0 {
        eprintln!("Invalid value for target SSIM quality");
    }

    if !args.output_dir.exists() {
        let _ = std::fs::create_dir(&args.output_dir);
    }

    if args.input.is_file() {
        let Some(image_filename) = args.input.file_name() else {
            return eprintln!("Couldn't read file path");
        };
        println!("Proccesing image: {}", image_filename.to_string_lossy());
        let image_processing = process_single_image(
            &args.input,
            args.ssim,
            &args.output_dir,
            args.image_format.as_str(),
        );
        match image_processing {
            Ok(_result) => println!("The image was successfully proccesed"),
            Err(e) => eprintln!("{e}"),
        }
    } else {
        let Some(dir_name) = args.input.file_name() else {
            return eprintln!("Couldn't read directory path");
        };
        println!("Proccesing directory {}", dir_name.to_string_lossy());
        let image_paths: Vec<PathBuf> = match std::fs::read_dir(args.input) {
            Ok(entries) => entries
                .filter_map(std::result::Result::ok)
                .map(|entry| entry.path())
                .collect(),
            Err(e) => {
                return eprintln!("{e}");
            }
        };
        let image_processing = process_images(
            &image_paths,
            args.ssim,
            &args.output_dir,
            args.image_format.as_str(),
        );
        match image_processing {
            Ok(_results) => println!("Images processed successfully"),
            Err(e) => eprintln!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::{NamedTempFile, tempdir};

    use super::*;
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    #[test]
    fn test_load_valid_image() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let test_image_path = Path::new("test_input2.png");

        let img = load_image(test_image_path)?;

        assert!(
            img.width() > 0 && img.height() > 0,
            "Imagen cargada debería de tener dimensiones positivas"
        );

        Ok(())
    }

    #[test]
    fn test_calculate_ssim_same_image() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let test_image_path = Path::new("test_input2.png");

        let img1 = load_image(test_image_path)?;
        let img2 = load_image(test_image_path)?;

        assert_eq!(
            img1.width(),
            img2.width(),
            "Las imágenes deben tener el mismo ancho para ser comparadas por SSIM"
        );
        assert_eq!(
            img1.height(),
            img2.height(),
            "Las imágenes deben tener la misma altura para ser comparadas por SSIM"
        );

        let ssim_result = calculate_ssim(&img1, &img2)?;

        let expected_ssim = 1.0;

        assert!(
            (ssim_result.score - expected_ssim).abs() < f64::EPSILON,
            "El SSIM de las imagen debería ser {} pero es {}",
            expected_ssim,
            ssim_result.score
        );

        Ok(())
    }

    #[test]
    fn test_save_image_as_jpeg() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let source_image_path = Path::new("test_input2.png");
        let img_to_save = load_image(source_image_path)?;

        let temp_output_file = NamedTempFile::with_suffix(".jpg")?;
        let output_path = temp_output_file.path();

        let quality = 85;

        save_image(&img_to_save, quality, output_path, "jpg", 10)?;

        assert!(
            output_path.exists(),
            "Salida debería de existir después de guardar"
        );

        let loaded_saved_img = load_image(output_path)?;

        assert!(
            loaded_saved_img.width() > 0 && loaded_saved_img.height() > 0,
            "La imagen guardada debe tener dimensiones positivas"
        );
        assert_eq!(
            loaded_saved_img.width(),
            img_to_save.width(),
            "La imagen guardada debe tener el mismo ancho a la original"
        );
        assert_eq!(
            loaded_saved_img.height(),
            img_to_save.height(),
            "La imagen guardada debe tener el mismo alto a la original"
        );

        Ok(())
    }

    #[test]
    fn test_save_image_as_webp() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let source_image_path = Path::new("test_input2.png");
        let img_to_save = load_image(source_image_path)?;

        let temp_output_file = NamedTempFile::with_suffix(".webp")?;
        let output_path = temp_output_file.path();

        let quality = 85;

        save_image(&img_to_save, quality, output_path, "webp", 10)?;

        assert!(
            output_path.exists(),
            "Salida debería de existir después de guardar"
        );

        let loaded_saved_img = load_image(output_path)?;

        assert!(
            loaded_saved_img.width() > 0 && loaded_saved_img.height() > 0,
            "La imagen guardada debe tener dimensiones positivas"
        );
        assert_eq!(
            loaded_saved_img.width(),
            img_to_save.width(),
            "La imagen guardada debe tener el mismo ancho a la original"
        );
        assert_eq!(
            loaded_saved_img.height(),
            img_to_save.height(),
            "La imagen guardada debe tener el mismo alto a la original"
        );

        Ok(())
    }

    #[test]
    fn test_save_image_as_avif() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let source_image_path = Path::new("test_input2.png");
        let img_to_save = load_image(source_image_path)?;

        let temp_output_file = NamedTempFile::with_suffix(".avif")?;
        let output_path = temp_output_file.path();

        let quality = 85;

        save_image(&img_to_save, quality, output_path, "avif", 10)?;

        assert!(
            output_path.exists(),
            "Salida debería de existir después de guardar"
        );

        println!("pron");
        let loaded_saved_img = load_image(output_path)?;
        println!("npor");

        assert!(
            loaded_saved_img.width() > 0 && loaded_saved_img.height() > 0,
            "La imagen guardada debe tener dimensiones positivas"
        );
        assert_eq!(
            loaded_saved_img.width(),
            img_to_save.width(),
            "La imagen guardada debe tener el mismo ancho a la original"
        );
        assert_eq!(
            loaded_saved_img.height(),
            img_to_save.height(),
            "La imagen guardada debe tener el mismo alto a la original"
        );

        Ok(())
    }

    #[test]
    fn test_get_ssim_for_jpg_quality() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let original_image_path = Path::new("test_input2.png");
        let original_img = load_image(original_image_path)?;

        let save_format = "jpg";
        let temp_output_file = NamedTempFile::with_suffix(format!(".{save_format}").as_str())?;
        let temp_output_path = temp_output_file.path();

        let quality_to_test = 100;

        let ssim_result = get_ssim_for_chosen_quality(
            &original_img,
            quality_to_test,
            temp_output_path,
            save_format,
        )?;

        let ssim_score = ssim_result.score;

        assert!(
            ssim_score < 1.0,
            "SSIM score should be less than 1.0 for compressed JPG (got {ssim_score})",
        );

        let high_quality_threshold = 0.95;
        assert!(
            ssim_score > high_quality_threshold,
            "SSIM score {ssim_score} should be greater than {high_quality_threshold}",
        );

        let loaded_saved_img_for_verification = load_image(temp_output_path)?;
        let ssim_manual = calculate_ssim(&original_img, &loaded_saved_img_for_verification)?.score;

        assert!(
            (ssim_score - ssim_manual).abs() < f64::EPSILON,
            "Function's SSIM score {ssim_score} should match manual calculation {ssim_manual}",
        );

        Ok(())
    }

    #[test]
    fn test_get_ssim_for_webp_quality() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let original_image_path = Path::new("test_input2.png");
        let original_img = load_image(original_image_path)?;

        let save_format = "webp";
        let temp_output_file = NamedTempFile::with_suffix(format!(".{save_format}").as_str())?;
        let temp_output_path = temp_output_file.path();

        let quality_to_test = 100;

        let ssim_result = get_ssim_for_chosen_quality(
            &original_img,
            quality_to_test,
            temp_output_path,
            save_format,
        )?;

        let ssim_score = ssim_result.score;

        assert!(
            ssim_score < 1.0,
            "SSIM score should be less than 1.0 for compressed JPG (got {ssim_score})",
        );
        let high_quality_threshold = 0.95;
        assert!(
            ssim_score > high_quality_threshold,
            "SSIM score {ssim_score} should be greater than {high_quality_threshold}",
        );

        let loaded_saved_img_for_verification = load_image(temp_output_path)?;
        let ssim_manual = calculate_ssim(&original_img, &loaded_saved_img_for_verification)?.score;

        assert!(
            (ssim_score - ssim_manual).abs() < f64::EPSILON,
            "Function's SSIM score {ssim_score} should match manual calculation {ssim_manual}",
        );

        Ok(())
    }

    #[test]
    fn test_get_ssim_for_avif_quality() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let original_image_path = Path::new("test_input2.png");
        let original_img = load_image(original_image_path)?;

        let save_format = "avif";
        let temp_output_file = NamedTempFile::with_suffix(format!(".{save_format}").as_str())?;
        let temp_output_path = temp_output_file.path();

        let quality_to_test = 100;

        let ssim_result = get_ssim_for_chosen_quality(
            &original_img,
            quality_to_test,
            temp_output_path,
            save_format,
        )?;

        let ssim_score = ssim_result.score;

        assert!(
            ssim_score < 1.0,
            "SSIM score should be less than 1.0 for compressed JPG (got {ssim_score})",
        );

        let high_quality_threshold = 0.95;
        assert!(
            ssim_score > high_quality_threshold,
            "SSIM score {ssim_score} should be greater than {high_quality_threshold}",
        );

        let loaded_saved_img_for_verification = load_image(temp_output_path)?;
        let ssim_manual = calculate_ssim(&original_img, &loaded_saved_img_for_verification)?.score;

        assert!(
            (ssim_score - ssim_manual).abs() < f64::EPSILON,
            "Function's SSIM score {ssim_score} should match manual calculation {ssim_manual}",
        );

        Ok(())
    }

    #[test]
    fn test_find_optimal_jpeg_quality() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let original_image_path = Path::new("test_input2.png");
        let original_img = load_image(original_image_path)?;

        let save_format = "jpg";

        let target_ssim = 0.98;

        let optimal_quality = find_optimal_image_quality(&original_img, target_ssim, save_format)?;

        let temp_output_file_verify = NamedTempFile::with_suffix(format!(".{save_format}"))?;
        let temp_output_path_verify = temp_output_file_verify.path();

        let actual_ssim_result = get_ssim_for_chosen_quality(
            &original_img,
            optimal_quality,
            temp_output_path_verify,
            save_format,
        )?;
        let actual_ssim_score = actual_ssim_result.score;

        let ssim_tolerance = 0.01;
        assert!(
            (actual_ssim_score - target_ssim).abs() < ssim_tolerance,
            "Found quality {optimal_quality} yields SSIM {actual_ssim_score} which is not close to target {target_ssim}",
        );

        Ok(())
    }

    #[test]
    fn test_find_optimal_webp_quality() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let original_image_path = Path::new("test_input2.png");
        let original_img = load_image(original_image_path)?;

        let save_format = "webp";

        let target_ssim = 0.98;

        let optimal_quality = find_optimal_image_quality(&original_img, target_ssim, save_format)?;

        let temp_output_file_verify = NamedTempFile::with_suffix(format!(".{save_format}"))?;
        let temp_output_path_verify = temp_output_file_verify.path();

        let actual_ssim_result = get_ssim_for_chosen_quality(
            &original_img,
            optimal_quality,
            temp_output_path_verify,
            save_format,
        )?;
        let actual_ssim_score = actual_ssim_result.score;

        let ssim_tolerance = 0.01;
        assert!(
            (actual_ssim_score - target_ssim).abs() < ssim_tolerance,
            "Found quality {optimal_quality} yields SSIM {actual_ssim_score} which is not close to target {target_ssim}",
        );

        Ok(())
    }

    #[test]
    fn test_find_optimal_avif_quality() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let original_image_path = Path::new("test_input2.png");
        let original_img = load_image(original_image_path)?;

        let save_format = "avif";

        let target_ssim = 0.98;

        let optimal_quality = find_optimal_image_quality(&original_img, target_ssim, save_format)?;

        let temp_output_file_verify = NamedTempFile::with_suffix(format!(".{save_format}"))?;
        let temp_output_path_verify = temp_output_file_verify.path();

        let actual_ssim_result = get_ssim_for_chosen_quality(
            &original_img,
            optimal_quality,
            temp_output_path_verify,
            save_format,
        )?;
        let actual_ssim_score = actual_ssim_result.score;

        let ssim_tolerance = 0.01;
        assert!(
            (actual_ssim_score - target_ssim).abs() < ssim_tolerance,
            "Found quality {optimal_quality} yields SSIM {actual_ssim_score} which is not close to target {target_ssim}",
        );

        Ok(())
    }

    #[test]
    fn test_process_images() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let temp_input_dir = tempdir()?;
        let temp_output_dir = tempdir()?;
        let output_dir_path = temp_output_dir.path();

        let source_image_path = Path::new("test_input2.png");
        let num_test_images = 3;
        let mut input_paths: Vec<PathBuf> = Vec::new();

        println!("Setting up test input files...");
        for i in 1..=num_test_images {
            let file_name = format!("test_image_{i}.png");
            let input_file_path = temp_input_dir.path().join(&file_name);
            println!(
                "Copying {} to {}...",
                source_image_path.display(),
                input_file_path.display()
            );
            std::fs::copy(source_image_path, &input_file_path)?;
            input_paths.push(input_file_path);
        }
        println!("Input files created.");

        let input_files_count = std::fs::read_dir(temp_input_dir.path())?.count();
        assert_eq!(
            input_files_count, num_test_images,
            "Incorrect number of input files created in temp directory"
        );

        let target_ssim = 0.98;

        let save_format = "jpg";
        println!("Calling process_images...");
        process_images(&input_paths, target_ssim, output_dir_path, save_format)?;
        println!("process_images finished.");

        let output_files: Vec<_> = std::fs::read_dir(output_dir_path)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .collect();

        println!("Checking output files in {}...", output_dir_path.display());
        assert_eq!(
            output_files.len(),
            num_test_images,
            "Should create one output file for each input file"
        );

        for output_file_path in output_files {
            println!("Checking output file: {}", output_file_path.display());
            assert!(
                output_file_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .ends_with(format!(".{save_format}").as_str()),
                "Output file should have .{save_format} extension",
            );
            assert!(
                fs::metadata(&output_file_path)?.len() > 0,
                "Output file should not be empty"
            );
        }

        Ok(())
    }
}
