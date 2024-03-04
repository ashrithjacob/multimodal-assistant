import * as readline from 'readline';
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function promptUser() {
    rl.question('Please enter something (type "exit" to quit): ', (input) => {
        if (input.toLowerCase() === 'exit') {
            rl.close();
        } else {
            console.log('You entered:', input);
            promptUser(); // Prompt again
        }
    });
}

promptUser(); // 