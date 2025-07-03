export function toPixels(
  value: string | number,
  element?: HTMLElement
): number {
  // 1. Numbers – React treats bare numbers as px
  if (typeof value === "number") return value;

  // 2. px strings
  if (value.endsWith("px")) return parseFloat(value);

  // 3. Viewport units
  const vhMatch = value.match(/([-+]?[0-9]*\.?[0-9]+)vh$/);
  if (vhMatch) return (parseFloat(vhMatch[1]) / 100) * window.innerHeight;

  const vwMatch = value.match(/([-+]?[0-9]*\.?[0-9]+)vw$/);
  if (vwMatch) return (parseFloat(vwMatch[1]) / 100) * window.innerWidth;

  const vminMatch = value.match(/([-+]?[0-9]*\.?[0-9]+)vmin$/);
  if (vminMatch) {
    const v = parseFloat(vminMatch[1]) / 100;
    return v * Math.min(window.innerWidth, window.innerHeight);
  }

  const vmaxMatch = value.match(/([-+]?[0-9]*\.?[0-9]+)vmax$/);
  if (vmaxMatch) {
    const v = parseFloat(vmaxMatch[1]) / 100;
    return v * Math.max(window.innerWidth, window.innerHeight);
  }

  // 4. Percentages – need parent’s computed style
  if (value.endsWith("%")) {
    if (!element || !element.parentElement) {
      throw new Error(
        "`toPixels`: percentage length requires an element with a parent."
      );
    }
    const parentPx = parseFloat(getComputedStyle(element.parentElement).height);
    return (parseFloat(value) / 100) * parentPx;
  }

  // 5. Fallback: read computed style once it’s rendered
  if (element) {
    return parseFloat(getComputedStyle(element).height);
  }

  throw new Error(`Unsupported CSS length: ${value}`);
}

/**
 * Compare two CSS lengths.
 *
 * @returns -1 if a < b, 1 if a > b, 0 if equal for the current layout.
 */
export function compareLengths(
  a: string | number,
  b: string | number,
  element: HTMLElement
): -1 | 0 | 1 {
  const pxA = toPixels(a, element);
  const pxB = toPixels(b, element);
  return pxA === pxB ? 0 : pxA < pxB ? -1 : 1;
}
