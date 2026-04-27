import { describe, expect, it } from "vitest";

import { enumLabel, statusTone } from "./utils";

describe("utils", () => {
  it("formats enum labels for operator copy", () => {
    expect(enumLabel("SHORT_TERM")).toBe("Short Term");
  });

  it("maps signal states to tones", () => {
    expect(statusTone("PREPARED")).toBe("pending");
    expect(statusTone("APPROVED")).toBe("success");
    expect(statusTone("UNMAPPABLE")).toBe("danger");
  });
});
