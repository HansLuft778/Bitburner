import { Colors } from "../lib.js";

export class Table {
    private header: string[];

    /**
     * array where each element is a row, each row is array of columns
     */
    private data: string[][];

    private colWidth: number[] = [];

    /**
     * Array of format [top left, top right, bottom left, bottom right, line char, t_down, t_up, t_left, t_right, cross]
     */
    private borderChars = {
        top_left: "╔",
        top_right: "╗",
        bottom_left: "╚",
        bottom_right: "╝",
        line_h: "═",
        line_v: "║",
        t_down: "╦",
        t_up: "╩",
        t_left: "╠",
        t_right: "╣",
        cross: "╬"
    };

    constructor(header: string[], data: string[][]) {
        this.header = header;
        this.data = data;

        if (data.length != 0 && header.length != data[0].length) {
            throw new Error("header and data must have same dimension");
        }

        this.updateColWidth();
    }

    private updateColWidth() {
        // find minimum width of each column
        let colWidth: number[] = Array(this.header.length).fill(0);
        for (let i = 0; i < this.header.length; i++) {
            colWidth[i] = Math.max(colWidth[i], this.header[i].length);
        }

        for (let i = 0; i < this.data.length; i++) {
            const row = this.data[i];
            for (let j = 0; j < row.length; j++) {
                const col = row[j];
                colWidth[j] = Math.max(colWidth[j], col.length);
            }
        }

        this.colWidth = colWidth;
    }

    private generateDeviderRow(
        leftBorderChar: string,
        lineChar: string,
        columnSeparatorChar: string,
        rightBorderChar: string
    ): string[] {
        let row: string[] = [leftBorderChar];
        for (let i = 0; i < this.header.length; i++) {
            row.push(
                Array(this.colWidth[i] + 2)
                    .fill(lineChar)
                    .join("")
            );
            if (i < this.header.length - 1) row.push(columnSeparatorChar);
        }
        row.push(rightBorderChar);

        return row;
    }

    private generateTextRow(
        columnSeperatorChar: string,
        rowData: string[]
    ): string[] {
        let row: string[] = [];
        for (let i = 0; i < rowData.length; i++) {
            let tmp = [
                columnSeperatorChar,
                " ",
                rowData[i],
                Array(Math.abs(this.colWidth[i] - rowData[i].length) + 1)
                    .fill(" ")
                    .join("")
            ];
            row.push(tmp.join(""));
        }

        row.push(columnSeperatorChar);

        return row;
    }

    private generateHeader(): string[] {
        let header: string[] = [];
        const first_row = this.generateDeviderRow(
            this.borderChars.top_left,
            this.borderChars.line_h,
            this.borderChars.t_down,
            this.borderChars.top_right
        );
        const seocnd_row = this.generateTextRow(
            this.borderChars.line_v,
            this.header
        );
        const third_row = this.generateDeviderRow(
            this.borderChars.t_left,
            this.borderChars.line_h,
            this.borderChars.cross,
            this.borderChars.t_right
        );

        header.push(first_row.join(""));
        header.push(seocnd_row.join(""));
        header.push(third_row.join(""));

        return header;
    }

    private generateBody(): string[] {
        let body: string[] = [];
        for (let data_idx = 0; data_idx < this.data.length; data_idx++) {
            const row = this.generateTextRow(
                this.borderChars.line_v,
                this.data[data_idx]
            );
            body.push(row.join(""));
        }

        return body;
    }

    private generateFooter() {
        // footer
        return this.generateDeviderRow(
            this.borderChars.bottom_left,
            this.borderChars.line_h,
            this.borderChars.t_up,
            this.borderChars.bottom_right
        ).join("");
    }

    /**
     * Prints the table to the terminal using the Netscript API
     * @param {NS} ns - The Netscript API object
     * @remarks
     * This method outputs the table in three parts:
     * 1. The header rows
     * 2. The body rows
     * 3. The footer row
     */
    public print(ns: NS) {
        const tableHeader = this.generateHeader();
        for (const row of tableHeader) {
            ns.print(row);
        }

        const tableBody = this.generateBody();
        for (const row of tableBody) {
            ns.print(row);
        }
        ns.print(this.generateFooter());
    }

    /**
     * @param symbol single character to use for table style, can be "=", "-", " "
     */
    public setBorderSymbol(symbol: string) {
        if (symbol == "=") {
            this.borderChars = {
                top_left: "╔",
                top_right: "╗",
                bottom_left: "╚",
                bottom_right: "╝",
                line_h: "═",
                line_v: "║",
                t_down: "╦",
                t_up: "╩",
                t_left: "╠",
                t_right: "╣",
                cross: "╬"
            };
        } else if (symbol == "-") {
            this.borderChars = {
                top_left: "┌",
                top_right: "┐",
                bottom_left: "└",
                bottom_right: "┘",
                line_h: "─",
                line_v: "│",
                t_down: "┬",
                t_up: "┴",
                t_left: "├",
                t_right: "┤",
                cross: "┼"
            };
        } else if (symbol == " ") {
            this.borderChars = {
                top_left: "",
                top_right: "",
                bottom_left: "",
                bottom_right: "",
                line_h: " ",
                line_v: "",
                t_down: " ",
                t_up: " ",
                t_left: "",
                t_right: "",
                cross: ""
            };
        } else {
            throw new Error('symbol must be element of ["=", "-"]');
        }
    }

    /**
     * Sets the characters used for drawing table borders and intersections
     * @param chars - An object containing border characters
     */
    public setBorderChars(chars: {
        top_left: string;
        top_right: string;
        bottom_left: string;
        bottom_right: string;
        line_h: string;
        line_v: string;
        t_down: string;
        t_up: string;
        t_left: string;
        t_right: string;
        cross: string;
    }) {
        this.borderChars = chars;
    }

    /**
     * Adds a new row of data to the table.
     * @param data An array of strings representing the new data row
     * @throws {Error} If the length of data array doesn't match the header length
     */
    public addRow(data: string[]) {
        if (data.length != this.header.length) {
            throw new Error("data dimenstion must match header");
        }
        this.data.push(data);
        this.updateColWidth();
    }
}

export async function main(ns: NS) {
    ns.tail();
    ns.clearLog();

    const header = ["symbol", "current", "min", "max", "forecast", "trend"];
    const data = [
        ["AERO", "14k", "12k", "23k", "0.555", "u"],
        ["APHE", "14k", "12k", "23k", "0.455", "d"],
        ["JGN", "1.422m", "65.270", "1.797m", "0.855", "u"],
        ["KGI", "62.8425kasdasdasdasd", "11.510k", "64.359k", "0.406", "↗"]
    ];

    const t = new Table(header, []);
    t.setBorderSymbol("=");
    t.addRow(["JGN", "1.422m", "65.270", "1.797m", "0.855", "u"]);
    t.addRow([
        "JGN",
        "1.422mkkkkkkkkkkkkkkkkkkkk",
        "65.270",
        "1.797m",
        "0.855",
        "u"
    ]);
    t.addRow(["a", "b", "", "", "", ""]);
    t.print(ns);
}
