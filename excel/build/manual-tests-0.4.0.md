# Manual tests for 0.4.0 (Windows, real Excel)

What the automated checks cannot do: dialogs, the ribbon, link prompts, a real
second install path, timing in a big workbook. Use **copies** of workbooks
only, in a scratch folder such as `C:\Users\<you>\extraplus-build\manual\`.

Before starting: install 0.4.0 with `install.bat` (Excel closed). In a new
workbook check `=EXTRAPLUS_VERSION()` = `0.4.0` and note
`=EXTRAPLUS_CACHE_STATS()`.

Timing in VBA (Alt+F11 > Ctrl+G, Immediate window): with
**Formulas > Calculation Options > Manual**, run
`t = Timer: Application.CalculateFull: Debug.Print Timer - t`
(full recalculation) or `t = Timer: Application.Calculate: Debug.Print Timer - t`
(only what changed). "Cold cache": close Excel, delete
`%LOCALAPPDATA%\Extraplus\resultcache\`, start Excel. "Warm": the same again
without deleting. Restore Calculation Options to Automatic afterwards.

Record for each test: result (pass/fail), what you saw, times.

## 1. Workbook saved with the add-in at another path

**1a. Your test workbook.** A copy of the real workbook (links to
`C:\Users\User\...` and another profile) opened on your PC:

1. Open the copy. Note whether a link prompt or a yellow security bar
   appears, and when (before/after the workbook is shown).
2. **Data > Edit Links**: every `extraplus.xlam` link should now point to
   `%LOCALAPPDATA%\Extraplus\extraplus.xlam` (yours). Note any that don't.
3. Values: spot-check 10 cells across 2P and 2S against the original
   (they must not be `#NAME?`).
4. Status bar: did Excel recalculate on open ("Calculating...")? How long?
   `=EXTRAPLUS_CACHE_STATS()` in an empty cell: misses = fits computed.
5. Close: does Excel ask to save? (Expected: yes, the links changed.)
   Save, reopen: Edit Links shows your path, no relink needed this time.

**1b. Two paths on one PC** (simulates two users):

1. Close Excel. Copy `%LOCALAPPDATA%\Extraplus\extraplus.xlam` to
   `C:\ExtraplusPathY\extraplus.xlam`.
2. Excel: File > Options > Add-ins > Manage: Excel Add-ins > Go: untick
   Extraplus, **Browse** to `C:\ExtraplusPathY\extraplus.xlam`, OK.
3. New workbook: x in A2:A13, y in B2:B13, a property in C2:C13;
   `=EXTRAP1_V(A2:A13,B2:B13)` in E2 (it becomes E2:E4),
   `=EXTRAP1_V(A2:A13,C2:C13,E4)` in F2, `=EXTRAP_ALL_V(A2:A13,B2:B13)` in H2.
   Save as `relink.xlsm`, close Excel.
4. Run `install.bat` again (it registers the add-in at the default path).
5. Open `relink.xlsm`: Edit Links should show the default path; values
   unchanged; no `#NAME?`.
6. Delete `C:\ExtraplusPathY`.

## 2. Insert a column in a workbook with many live formulas

On a copy of the test workbook (after test 1a, links fixed):

1. Cold cache: right-click column BV > Insert. Time it (stopwatch, or Manual
   mode + `Application.Calculate` timing). Excel should not freeze; check
   that no cells next to the function blocks were cleared or overwritten
   (compare 5 blocks and their neighbours before/after).
2. Undo, then insert again with a warm cache. Time it.
3. Delete the inserted column. Same checks.

| | cold | warm |
|---|---|---|
| insert column at BV (s) | | |
| CalculateFull (s) | | |

## 3. Reopen a large workbook after saving

1. Save the copy from test 2, close Excel, reopen it. Note whether Excel
   recalculates on open, how long it takes, and whether any values changed
   (spot-check 10 cells).
2. The same with the cache file deleted first (cold).

## 4. Extraplus > Write values

On a copy (saved, in a local folder):

1. The ribbon has an **Extraplus** tab with **Write values**.
2. Select one function block (e.g. 3 cells of an `EXTRAP1_V`) > Write
   values: a confirmation, then a summary. The block is now numbers; its
   values equal the formula's; the backup file
   `<name> (before Extraplus values <time>).xlsm` exists next to the
   workbook; the sheet **Extraplus values** lists the block.
3. Select one cell on 2P > Write values (whole sheet). Time it. Check:
   - chained blocks (third argument = B of another block) were written,
     with the same values as before;
   - blocks that showed errors are still formulas and listed;
   - nothing else on the sheet changed (formulas that are not Extraplus).
4. Open the backup copy: it still has the formulas.
5. In an **unsaved** new workbook: Write values says to save first and
   changes nothing.
6. Cancel at the confirmation: nothing changes, no backup file.

## 5. The add-in absent

1. Excel: untick Extraplus in the Add-ins dialog (or run the uninstaller on a
   test account). Open a copy of a workbook with Extraplus formulas.
2. Note what Excel shows (link prompt, `#NAME?` after recalculation).
   **Don't save** (the save warning needs the add-in, so it cannot appear).
3. Re-tick the add-in (or reinstall), reopen the copy: values back.

## 6. Save warning

1. With the add-in loaded, in a test workbook enter
   `=EXTRAP1(A2:A13,B2:B13)+#NAME?`: an Extraplus formula that shows
   `#NAME?`, as it would when the add-in is not found.
2. Ctrl+S: a warning with the number of cells and an example address;
   **No** (the default) cancels the save, **Yes** saves.
3. A workbook without such cells saves without a warning.

## 7. Errors keep the 3-cell shape

1. `=EXTRAP1_V(A2:A3,B2:B3)` in a 3-cell array (two points: an error): all 3
   cells show `Error: At least three paired numeric x/y observations are
   required.`; the array stays 3 cells.
2. A formula using its third cell as `b_fixed` shows
   `Error: b_fixed must be a number.` (not a number).
3. Put `=NA()` in a cell and use it as `b_fixed`: `Error: b_fixed must be a
   number.`

## 8. Cache switch

1. Close Excel, `setx EXTRAPLUS_CACHE off`, start Excel:
   `=EXTRAPLUS_CACHE_STATS()` shows `cache off`; the functions still work.
2. `setx EXTRAPLUS_CACHE on`, restart Excel: `cache on`.
3. Two Excel windows (two instances: start the second with Alt held, answer
   Yes) on two workbooks at once: both calculate; no errors.
