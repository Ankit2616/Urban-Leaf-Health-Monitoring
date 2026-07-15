const fs = require('fs');
const path = require('path');

const ROOT = process.cwd();
const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp']);

function walkDir(dir, baseDir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      // Skip Jupyter checkpoint dirs
      if (entry.name === '.ipynb_checkpoints') continue;
      files.push(...walkDir(fullPath, baseDir));
    } else if (IMAGE_EXT.has(path.extname(entry.name).toLowerCase())) {
      // Store path relative to the R2 bucket root (same as project root)
      files.push(path.relative(baseDir, fullPath));
    }
  }
  return files;
}

// These correspond to the base directories as they'll appear in the R2 bucket
const imageSources = [
  {
    name: 'event_jpeg',
    dir: path.join(ROOT, 'data/02_comparison_based_on_events/event_1/tiff_to_jpeg'),
    r2Prefix: 'data/02_comparison_based_on_events/event_1/tiff_to_jpeg',
  },
  {
    name: 'sample_rgb',
    dir: path.join(ROOT, 'data/01_area_of_interest_selection_using_sampling'),
    r2Prefix: 'data/01_area_of_interest_selection_using_sampling',
  },
  {
    name: 'asset',
    dir: path.join(ROOT, 'assets'),
    r2Prefix: 'assets',
  },
];

const manifest = {
  csvFiles: [
    'data/02_comparison_based_on_events/event_1/csv_files/cleaned_images_features.csv',
    'data/03_hasdeo_1000_all_bands/metadata/export_log.csv',
  ],
};

for (const source of imageSources) {
  if (fs.existsSync(source.dir)) {
    // Walk the directory, then prefix with the R2 bucket path
    const relativeFiles = walkDir(source.dir, source.dir);
    manifest[source.name] = relativeFiles.map(f => `${source.r2Prefix}/${f}`);
    console.log(`- ${source.name}: ${manifest[source.name].length} files`);
  } else {
    manifest[source.name] = [];
    console.log(`- ${source.name}: 0 files (directory missing)`);
  }
}

// Also scan model result dirs for masks/overlays
const modelDirs = [
  path.join(ROOT, 'results'),
  path.join(ROOT, 'h100_config/results'),
];
const masks = [];
const overlays = [];
const summaries = [];
for (const dir of modelDirs) {
  if (fs.existsSync(dir)) {
    const files = walkDir(dir, ROOT);
    for (const f of files) {
      if (f.includes('_mask.png')) masks.push(f);
      else if (f.includes('_overlay.png')) overlays.push(f);
      else if (f.includes('summary.json')) summaries.push(f);
    }
  }
}
manifest.models = { masks, overlays, summaries };

fs.writeFileSync(path.join(ROOT, 'r2-manifest.json'), JSON.stringify(manifest, null, 2));
console.log('\nManifest generated at r2-manifest.json');
console.log(`Total images: ${(manifest.event_jpeg || []).length + (manifest.sample_rgb || []).length + (manifest.asset || []).length}`);