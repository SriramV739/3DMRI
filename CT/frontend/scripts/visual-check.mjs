import { chromium } from '@playwright/test';
import { PNG } from 'pngjs';

const TARGET = process.env.VISUAL_CHECK_URL || 'http://127.0.0.1:5173';

function countBrightPixels(buffer) {
  const png = PNG.sync.read(buffer);
  let bright = 0;
  let varied = 0;
  for (let index = 0; index < png.data.length; index += 4) {
    const red = png.data[index];
    const green = png.data[index + 1];
    const blue = png.data[index + 2];
    const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    if (luminance > 45) bright += 1;
    if (Math.max(red, green, blue) - Math.min(red, green, blue) > 12) varied += 1;
  }
  return { bright, varied, total: png.width * png.height };
}

async function checkViewport(browser, name, viewport) {
  const page = await browser.newPage({ viewport });
  await page.goto(TARGET, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('canvas', { state: 'visible', timeout: 60000 });
  await page.waitForTimeout(12000);

  const canvas = page.locator('canvas').first();
  const box = await canvas.boundingBox();
  if (!box || box.width < 100 || box.height < 100) {
    throw new Error(`${name}: canvas is missing or too small`);
  }

  const image = await page.screenshot({
    clip: {
      x: Math.max(0, box.x),
      y: Math.max(0, box.y),
      width: Math.min(box.width, viewport.width),
      height: Math.min(box.height, viewport.height),
    },
  });
  const counts = countBrightPixels(image);
  const brightRatio = counts.bright / counts.total;
  const variedRatio = counts.varied / counts.total;

  if (brightRatio < 0.015 || variedRatio < 0.015) {
    throw new Error(`${name}: canvas appears blank; bright=${brightRatio.toFixed(4)} varied=${variedRatio.toFixed(4)}`);
  }

  console.log(`VISUAL_OK ${name} ${viewport.width}x${viewport.height} bright=${brightRatio.toFixed(3)} varied=${variedRatio.toFixed(3)}`);
  await page.close();
}

const browser = await chromium.launch({ headless: true });
try {
  await checkViewport(browser, 'desktop', { width: 1440, height: 900 });
  await checkViewport(browser, 'mobile', { width: 390, height: 844, isMobile: true });
} finally {
  await browser.close();
}
