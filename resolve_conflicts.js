const fs = require('fs');
const glob = require('glob');

function resolveConflictsHead(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Regular expression to match git merge conflict blocks
    // Format: 
    // <<<<<<< HEAD
    // (content we want to keep)
    // =======
    // (content we want to discard)
    // >>>>>>> (commit hash)
    
    const conflictRegex = /<<<<<<< HEAD\n([\s\S]*?)=======\n[\s\S]*?>>>>>>> [0-9a-fA-F]+/g;
    
    if (conflictRegex.test(content)) {
        console.log(`Resolving conflicts in ${filePath}...`);
        content = content.replace(conflictRegex, '$1');
        fs.writeFileSync(filePath, content, 'utf8');
        return true;
    }
    return false;
}

// Find all files that might have conflicts
const files = [
    'public/dashboard.js',
    'public/index.html',
    'run_pipeline.py',
    'upload_handler.py'
];

let modified = false;
for (const file of files) {
    if (fs.existsSync(file)) {
        if (resolveConflictsHead(file)) {
            modified = true;
        }
    }
}

if (!modified) {
    console.log("No merge conflicts found to resolve.");
}
