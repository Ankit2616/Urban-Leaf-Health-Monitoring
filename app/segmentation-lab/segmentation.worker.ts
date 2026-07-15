type ClassKey = 0 | 1 | 2 | 3 | 4;

type ClassSummary = {
  id: ClassKey;
  name: string;
  color: string;
  pixels: number;
  ratio: number;
};

const CLASS_META: Array<{ id: ClassKey; name: string; color: [number, number, number] }> = [
  { id: 0, name: "Vegetation", color: [27, 128, 69] },
  { id: 1, name: "Sparse Vegetation", color: [166, 217, 106] },
  { id: 2, name: "Bare Soil / Rock", color: [217, 172, 84] },
  { id: 3, name: "Built-up / Urban", color: [120, 120, 120] },
  { id: 4, name: "Water / Shadow", color: [43, 131, 186] }
];

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function rgbToHsv(r: number, g: number, b: number) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  let h = 0;

  if (delta) {
    if (max === rn) h = ((gn - bn) / delta) % 6;
    else if (max === gn) h = (bn - rn) / delta + 2;
    else h = (rn - gn) / delta + 4;
  }

  return {
    h: (h * 60 + 360) % 360,
    s: max === 0 ? 0 : delta / max,
    v: max
  };
}

function classifyReference(r: number, g: number, b: number): ClassKey {
  const { h, s, v } = rgbToHsv(r, g, b);
  const exg = (2 * g - r - b) / 255;
  const blueDominance = (b - Math.max(r, g)) / 255;
  const warmth = (r - b) / 255;

  if (v < 0.2 || (blueDominance > 0.08 && s < 0.45)) return 4;
  if (exg > 0.16 && h >= 65 && h <= 165) return v > 0.46 ? 0 : 1;
  if (s < 0.16 && v > 0.42) return 3;
  if (warmth > 0.1 && v > 0.35) return 2;
  if (h >= 75 && h <= 150) return 1;
  return v > 0.58 ? 3 : 2;
}

function classifyProject(r: number, g: number, b: number): ClassKey {
  const { h, s, v } = rgbToHsv(r, g, b);
  const greenBoost = (g - (r + b) / 2) / 255;
  const dryness = (r - g) / 255 + (Math.abs(r - b) / 255) * 0.4;
  const waterCue = (b - r) / 255;
  const brightNeutral = s < 0.14 && v > 0.48;

  if (v < 0.18 || (waterCue > 0.09 && s < 0.52)) return 4;
  if (greenBoost > 0.12 && h >= 68 && h <= 150) return v > 0.42 ? 0 : 1;
  if (dryness > 0.14 && v > 0.28) return 2;
  if (brightNeutral || (s < 0.22 && v > 0.36)) return 3;
  if (h >= 70 && h <= 145) return 1;
  return v > 0.52 ? 3 : 2;
}

function smoothMask(mask: Uint8Array, width: number, height: number) {
  const output = new Uint8Array(mask);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const counts = [0, 0, 0, 0, 0];
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          counts[mask[(y + ky) * width + (x + kx)]] += 1;
        }
      }
      let best: ClassKey = 0;
      let bestCount = -1;
      for (let index = 0; index < counts.length; index += 1) {
        if (counts[index] > bestCount) {
          best = index as ClassKey;
          bestCount = counts[index];
        }
      }
      output[y * width + x] = best;
    }
  }
  return output;
}

function summarizeMask(mask: Uint8Array): ClassSummary[] {
  const total = mask.length || 1;
  return CLASS_META.map((item) => {
    let pixels = 0;
    for (let index = 0; index < mask.length; index += 1) {
      if (mask[index] === item.id) pixels += 1;
    }
    return {
      id: item.id,
      name: item.name,
      color: `rgb(${item.color.join(", ")})`,
      pixels,
      ratio: pixels / total
    };
  });
}

self.onmessage = (event: MessageEvent<{ imageDataArray: Uint8ClampedArray; width: number; height: number }>) => {
  const { imageDataArray, width, height } = event.data;
  
  const referenceMask = new Uint8Array(width * height);
  const projectMask = new Uint8Array(width * height);

  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    const r = imageDataArray[offset];
    const g = imageDataArray[offset + 1];
    const b = imageDataArray[offset + 2];
    referenceMask[index] = classifyReference(r, g, b);
    projectMask[index] = classifyProject(r, g, b);
  }

  const referenceSmoothed = smoothMask(referenceMask, width, height);
  const projectSmoothed = smoothMask(projectMask, width, height);
  let agreement = 0;

  for (let index = 0; index < referenceSmoothed.length; index += 1) {
    if (referenceSmoothed[index] === projectSmoothed[index]) agreement += 1;
  }

  const referenceSummary = summarizeMask(referenceSmoothed);
  const projectSummary = summarizeMask(projectSmoothed);
  const canopyCover = clamp01(projectSummary[0].ratio + projectSummary[1].ratio) * 100;
  const builtUpShare = projectSummary[3].ratio * 100;
  const exposedSurface = (projectSummary[2].ratio + projectSummary[3].ratio) * 100;
  const waterShadowShare = projectSummary[4].ratio * 100;

  const agreementPct = (agreement / referenceSmoothed.length) * 100;

  (self as any).postMessage(
    {
      referenceSmoothed,
      projectSmoothed,
      referenceSummary,
      projectSummary,
      canopyCover,
      builtUpShare,
      exposedSurface,
      waterShadowShare,
      agreement: agreementPct
    },
    // Transfer the typed arrays to avoid copying memory
    [referenceSmoothed.buffer, projectSmoothed.buffer]
  );
};
