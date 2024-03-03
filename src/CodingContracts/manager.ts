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
                    break;
                case "Subarray with Maximum Sum":
                    subarraywithMaximumSum(ns, contract, server);
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
                    algorithmicStockTraderII(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader III":
                    algorithmicStockTraderIII(ns, contract, server);
                    break;
                case "Algorithmic Stock Trader IV":
                    break;
                case "Minimum Path Sum in a Triangle":
                    minimumPathSumInATriangle(ns, contract, server);
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
                    findAllValidMathExpressions(ns, contract, server);
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
                    throw new Error("unknown contract type: " + contractType);
            }
        }
    }
}

export function findShortestPath(ns: NS, contract: string, server: string) {
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
        ns.tprint(Colors.RED + `failed to solve contract ${contract} on server ${server}`);
        throw new Error("failed to solve contract");
    }

    ns.print(Colors.GREEN + success);
}

export function totalWaysToSum(ns: NS, contract: string, server: string) {
    const data: number = ns.codingcontract.getData(contract, server);

    if (typeof data !== "number" || data < 1) {
        ns.tprint(Colors.RED + "invalid data for contract " + contract + " on server " + server);
        return;
    }

    function sumCombinations(target: number, current: number[] = [], start = 1, result: number[][] = []) {
        if (target === 0) {
            result.push(current.slice()); // Push a copy of the current combination to the result
            return;
        }

        for (let i = start; i <= target; i++) {
            current.push(i);
            sumCombinations(target - i, current, i, result);
            current.pop();
        }

        return result;
    }

    ns.print(sumCombinations(data));

    const res = sumCombinations(data)?.filter((arr) => arr.length !== 1).length;
    if (res === undefined) {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }
    const success = ns.codingcontract.attempt(res, contract, server);

    if (success === "") {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    ns.print(Colors.GREEN + success);
}

function findAllValidMathExpressions(ns: NS, contract: string, server: string) {
    const data = ns.codingcontract.getData(contract, server);

    function opStr(mu: number, len: number) {
        const ops = ["", "-", "+", "*"];
        const s: string[] = [];
        while (mu >= 4) {
            s.push(ops[mu % 4]);
            mu -= mu % 4;
            mu /= 4;
        }
        s.push(ops[mu]);
        while (s.length < len) {
            s.push(ops[0]);
        }
        return s;
    }
    // ["012345", N], where N is the assertion
    // each gap between two digits can be one of 4 things (blank, -, +, *),
    // so there are 4^d permutations, where d=length-1
    const answers = [];
    const digits = data[0];
    const assertion = data[1];

    const permutations = Math.pow(4, digits.length - 1);
    for (let i = 0; i < permutations; i++) {
        // turn the permutation number into a list of operators
        const ops = opStr(i, digits.length - 1);
        // interleave digits and ops
        let expr = "";
        for (let j = 0; j < ops.length; j++) {
            expr += digits[j] + ops[j];
        }
        expr += digits[ops.length];
        // leading 0s sometimes throw an error about octals
        try {
            if (eval(expr) == assertion) {
                answers.push(expr);
            }
        } catch (e) {
            //
        }
    }

    const success = ns.codingcontract.attempt(answers, contract, server);

    if (success === "") {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    ns.print(Colors.GREEN + success);
}

function algorithmicStockTraderII(ns: NS, contract: string, server: string) {
    const stockPrice: number[] = ns.codingcontract.getData(contract, server);

    stockPrice.push(0);

    let profit = 0;
    for (let i = 0; i < stockPrice.length; i++) {
        if (stockPrice[i] < stockPrice[i + 1]) {
            profit += stockPrice[i + 1] - stockPrice[i];
        }
    }

    profit = Math.max(profit, 0);

    const success = ns.codingcontract.attempt(profit, contract, server);

    if (success === "") {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    ns.print(Colors.GREEN + success + " from " + contract);
}

export function algorithmicStockTraderIII(ns: NS, contract: string, server: string) {
    const stockPrice: number[] = [
        11, 76, 116, 182, 56, 33, 13, 149, 69, 150, 88, 15, 45, 117, 142, 80, 78, 150, 50, 5, 104, 79, 23, 21, 107, 38,
        54, 181,
    ]; //ns.codingcontract.getData(contract, server);

    stockPrice.push(-1);

    // 11,76,116,182,56,33,13,149,69,150,88,15,45,117,142,80,78,150,50,5,104,79,23,21,107,38,54,181
    //     171               136    81           127           72       99          86        134
    const maximums: number[] = [];

    let profit = 0;
    for (let i = 0; i < stockPrice.length; i++) {
        if (stockPrice[i] < stockPrice[i + 1]) {
            profit += stockPrice[i + 1] - stockPrice[i];
        }
        if (stockPrice[i] > stockPrice[i + 1]) {
            maximums.push(profit);
            profit = 0;
        }
    }

    // best top 2 profits
    maximums.sort((a, b) => b - a);
    const best = maximums.slice(0, 2);
    const sum = best.reduce((a, b) => a + b, 0);
    ns.print(Colors.E_ORANGE + sum + " | " + best + " | " + maximums);

    stockPrice.pop();
    maximums.length = 0;
    // second method
    // find two lowest numbers in stockprice, and add them to maximums
    const min = Math.min(...stockPrice);
    const minIdx = stockPrice.indexOf(min);

    const secMin = Math.min(...stockPrice.filter((p) => p !== min));
    const secMinIdx = stockPrice.indexOf(secMin);

    const max = Math.max(...stockPrice);
    const maxIdx = stockPrice.indexOf(max);

    const secMax = Math.max(...stockPrice.filter((p) => p !== max));
    const secMaxIdx = stockPrice.indexOf(secMax);

    ns.print(min + " " + minIdx + "|" + secMin + " " + secMinIdx);
    ns.print(max + " " + maxIdx + "|" + secMax + " " + secMaxIdx);

    if (minIdx < secMinIdx) {
        if (maxIdx < secMaxIdx) {
            // match min with max
            maximums.push(max - min);
            maximums.push(secMax - secMin);
            ns.print("here");
        } else {
            // match min with max2
            maximums.push(secMax - min);
            maximums.push(max - secMin);
            ns.print("here2");
        }
    } else {
        if (maxIdx < secMaxIdx) {
            // match min2 with max
            maximums.push(secMax - min);
            maximums.push(max - secMin);
            ns.print("here3");
        } else {
            // match min2 with max2
            maximums.push(secMax - secMin);
            maximums.push(max - min);
            ns.print("here4");
        }
    }
    const sum2 = maximums.reduce((a, b) => a + b, 0);

    ns.print(Colors.E_ORANGE + maximums + " | " + sum2);

    const success = ns.codingcontract.attempt(sum, contract, server);

    if (success === "") {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    ns.print(Colors.GREEN + success + " from " + contract);
}

function minimumPathSumInATriangle(ns: NS, contract: string, server: string) {
    const triangle: number[][] = ns.codingcontract.getData(contract, server);

    if (triangle.length === 0) {
        ns.tprint(Colors.RED + "invalid data for contract " + contract + " on server " + server);
        return;
    }

    let pathSum = 0;
    pathSum += triangle[0][0];
    let currentCol = 0;
    for (let i = 0; i < triangle.length; i++) {
        if (triangle[i + 1] === undefined) break;
        const possibleMoves = [triangle[i + 1][currentCol], triangle[i + 1][currentCol + 1]];

        if (possibleMoves[0] > possibleMoves[1]) currentCol = currentCol + 1;
        const move = Math.min(...possibleMoves);
        pathSum += move;
    }

    const success = ns.codingcontract.attempt(pathSum, contract, server);

    if (success === "") {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    ns.print(Colors.GREEN + success);
}

function subarraywithMaximumSum(ns: NS, contract: string, server: string) {
    const data: number[] = ns.codingcontract.getData(contract, server);

    if (data.length === 0) {
        ns.tprint(Colors.RED + "invalid data for contract " + contract + " on server " + server);
        return;
    }

    let maxSum = -Infinity;
    let currentSum = 0;

    for (let i = 0; i < data.length; i++) {
        currentSum = Math.max(data[i], currentSum + data[i]);
        maxSum = Math.max(maxSum, currentSum);
    }

    const success = ns.codingcontract.attempt(maxSum, contract, server);

    if (success === "") {
        ns.tprint(Colors.RED + "failed to solve contract " + contract + " on server " + server);
        return;
    }

    ns.print(Colors.GREEN + success);
}
