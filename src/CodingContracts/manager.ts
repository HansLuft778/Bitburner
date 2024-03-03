import { Colors, serverScanner } from "@/lib";
import { NS } from "@ns";
export async function main(ns: NS) {
    ns.tail();
    ns.disableLog("ALL");
    const cc = ns.codingcontract;

    const allServers = serverScanner(ns);

    for (const server of allServers) {
        const contracts = ns.ls(server).filter((p) => p.includes("cct"));

        if (contracts.length === 0) continue;

        ns.print(`server ${server} has ${contracts.length} contracts`);

        for (const contract of contracts) {
            const contractType = cc.getContractType(contract, server);
            ns.print(`contract ${contract} is of type ${contractType}`);

            switch (contractType) {
                case "Find Largest Prime Factor":
                    //
                    break;
                case "Subarray with Maximum Sum":
                    break;
                case "Total Ways to Sum":
                    totalWaysToSum(ns, contract, server);
                    break;
                case "Total Ways to Sum II":
                    break;
                case "Spiralize Matrix":
                    break;
                case "Array Jumping Game":
                    break;
                case "Array Jumping Game II":
                    break;
                case "Merge Overlapping Intervals":
                    break;
                case "Generate IP Addresses":
                    break;
                case "Algorithmic Stock Trader I":
                    break;
                case "Algorithmic Stock Trader II":
                    break;
                case "Algorithmic Stock Trader III":
                    break;
                case "Algorithmic Stock Trader IV":
                    break;
                case "Minimum Path Sum in a Triangle":
                    break;
                case "Unique Paths in a Grid I":
                    break;
                case "Unique Paths in a Grid II":
                    break;
                case "Shortest Path in a Grid":
                    findShortestPath(ns, contract, server);
                    break;
                case "Sanitize Parentheses in Expression":
                    break;
                case "Find All Valid Math Expressions":
                    break;
                case "HammingCodes: Integer to Encoded Binary":
                    break;
                case "HammingCodes: Encoded Binary to Integer":
                    break;
                case "Proper 2-Coloring of a Graph":
                    break;
                case "Compression I: RLE Compression":
                    break;
                case "Compression II: LZ Decompression":
                    break;
                case "Compression III: LZ Compression":
                    break;
                case "Encryption I: Caesar Cipher":
                    break;
                case "Encryption II: Vigenère Cipher":
                    break;
                default:
                    break;
            }
        }
    }
}

function generateSwitchCase(ns: NS) {
    let switchCase = "";
    switchCase += "switch (contractType) {\n";
    for (const contractType of ns.codingcontract.getContractTypes()) {
        switchCase += `case "${contractType}":\n`;
        switchCase += `break;\n`;
    }
    switchCase += "default:\n";
    switchCase += "break;\n";
    switchCase += "}\n";
    return switchCase;
}

function findShortestPath(ns: NS, contract: string, server: string) {
    const data: number[][] = ns.codingcontract.getData(contract, server);

    if (data.length === 0 || data[0][0] === 1 || data[data.length - 1][data[0].length - 1] === 1) {
        ns.print("no solution found");
        return "";
    }

    const visited: boolean[][] = Array.from({ length: data.length }, () => Array(data[0].length).fill(false));
    const moves: string[] = [""];
    const min = { length: Number.MAX_SAFE_INTEGER };
    const currentPath: string[] = [];

    const startX = 0;
    const startY = 0;
    const endX = data.length - 1;
    const endY = data[0].length - 1;

    const directions = {
        R: [0, 1],
        L: [0, -1],
        D: [1, 0],
        U: [-1, 0],
    };

    function findPath(
        grid: number[][],
        visited: boolean[][],
        i: number,
        j: number,
        x: number,
        y: number,
        moves: string[],
        min: { length: number },
        currentPath: string[],
        directions: Record<string, number[]>,
    ) {
        if (i == x && j == y) {
            if (currentPath.length < min.length) {
                min.length = currentPath.length;
                moves[0] = currentPath.join("");
            }
            return;
        }

        visited[i][j] = true;

        for (const dir in directions) {
            const [row, col] = directions[dir];

            if (isValid(grid, visited, i + row, j + col)) {
                currentPath.push(dir);
                findPath(grid, visited, i + row, j + col, x, y, moves, min, currentPath, directions);
                currentPath.pop();
            }
        }

        visited[i][j] = false;
    }

    function isValid(grid: number[][], visited: boolean[][], row: number, col: number): boolean {
        return (
            row >= 0 && row < grid.length && col >= 0 && col < grid[0].length && !visited[row][col] && !grid[row][col]
        );
    }

    findPath(data, visited, startX, startY, endX, endY, moves, min, currentPath, directions);

    ns.print(moves[0]);

    const success = ns.codingcontract.attempt(moves[0], contract, server);
    if (success === "") {
        ns.tprint(`failed to solve contract ${contract} on server ${server}`);
        throw new Error("failed to solve contract");
    }

    ns.print(Colors.GREEN + success);
}

function totalWaysToSum(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);

    // 64
    ns.print(data);

    /**
    
    1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1
    1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+2
    1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+3
    1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+4
    1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+5
    ...
    1+63

    => 63 -> n - 1


    2+62
    3+61
    4+60
    5+59
    ...
    31+33
    32+32

    => floor(n / 2) - 1



    for 7:
    1+1+1+1+1+1+1
    1+1+1+1+1+2
    1+1+1+1+3
    1+1+1+4
    1+1+5
    1+6
     => n-1

    2+2+2+1
    2+2+3
    2+5
     => floor(n / 2) - 1

    3+3+1
    3+4

    2+5
    3+4
     => floor(n / 2) - 1

    ----------

    1+
    2+
    3+
    4+
    5+
    6+

     */

    function sumCombinations(target: number, current: number[] = [], start = 1, result: number[][] = []) {
        if (target === 0) {
            result.push(current.slice()); // Push a copy of the current combination to the result
            return;
        }

        for (let i = start; i <= target; i++) {
            current.push(i);
            sumCombinations(target - i, current, i, result); // Recursive call with updated target and current combination
            current.pop(); // Backtrack by removing the last element from the current combination
        }

        return result;
    }

    const found: Set<string> = new Set();

    // for (let i = 1; i <= 7; i++) {
    //     const gorg = 7 - i;
    //     found.add([i, gorg].sort().join(","));

    //     // can i be split into two numbers?
    //     if (i > 2) {
    //         const half = Math.floor(i / 2);
    //         found.add([half, i - half].sort().join(","));
    //     }
    // }

    ns.print(sumCombinations(7));
}
