"""
Sample Pack Generator and Validator for Comparative Analysis

This script helps create, validate, and manage sample data packs
for the comparative analysis feature.
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path

class SamplePackGenerator:
    """
    Utility class for managing comparative analysis sample packs
    """
    
    def __init__(self, samples_file: str = "comparative_samples.json"):
        self.samples_file = Path(__file__).parent / samples_file
        self.samples_data = self._load_samples()
    
    def _load_samples(self) -> Dict[str, Any]:
        """Load existing sample data from JSON file"""
        try:
            if self.samples_file.exists():
                with open(self.samples_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"sample_packs": {}}
        except Exception as e:
            print(f"Error loading samples: {e}")
            return {"sample_packs": {}}
    
    def save_samples(self) -> bool:
        """Save sample data to JSON file"""
        try:
            with open(self.samples_file, 'w', encoding='utf-8') as f:
                json.dump(self.samples_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving samples: {e}")
            return False
    
    def add_sample_pack(self, pack_id: str, name: str, description: str, samples: List[Dict[str, str]]) -> bool:
        """
        Add a new sample pack
        
        Args:
            pack_id: Unique identifier for the pack
            name: Display name with emoji
            description: Brief description of the pack
            samples: List of sample dictionaries with 'label' and 'text' keys
        """
        if not samples or len(samples) < 2:
            print("Error: Sample pack must contain at least 2 samples")
            return False
        
        for sample in samples:
            if 'label' not in sample or 'text' not in sample:
                print("Error: Each sample must have 'label' and 'text' keys")
                return False
            
            if len(sample['text'].strip()) < 10:
                print(f"Warning: Sample '{sample['label']}' has very short text")
        
        self.samples_data["sample_packs"][pack_id] = {
            "name": name,
            "description": description,
            "samples": samples
        }
        
        return self.save_samples()
    
    def validate_samples(self) -> List[str]:
        """Validate all sample packs and return list of issues"""
        issues = []
        
        if "sample_packs" not in self.samples_data:
            issues.append("No sample_packs key found in data")
            return issues
        
        for pack_id, pack_data in self.samples_data["sample_packs"].items():
            # Check pack structure
            required_keys = ["name", "description", "samples"]
            for key in required_keys:
                if key not in pack_data:
                    issues.append(f"Pack '{pack_id}' missing required key: {key}")
            
            # Check samples
            if "samples" in pack_data:
                samples = pack_data["samples"]
                
                if len(samples) < 2:
                    issues.append(f"Pack '{pack_id}' has fewer than 2 samples")
                
                if len(samples) > 5:
                    issues.append(f"Pack '{pack_id}' has more than 5 samples (may affect UI)")
                
                for i, sample in enumerate(samples):
                    if "label" not in sample:
                        issues.append(f"Pack '{pack_id}' sample {i} missing 'label'")
                    
                    if "text" not in sample:
                        issues.append(f"Pack '{pack_id}' sample {i} missing 'text'")
                    
                    if "text" in sample:
                        text_len = len(sample["text"].strip())
                        if text_len < 10:
                            issues.append(f"Pack '{pack_id}' sample '{sample.get('label', i)}' text too short ({text_len} chars)")
                        elif text_len > 1000:
                            issues.append(f"Pack '{pack_id}' sample '{sample.get('label', i)}' text very long ({text_len} chars)")
        
        return issues
    
    def get_sample_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sample packs"""
        stats = {
            "total_packs": 0,
            "total_samples": 0,
            "avg_samples_per_pack": 0,
            "text_length_stats": {
                "min": float('inf'),
                "max": 0,
                "avg": 0
            },
            "pack_details": {}
        }
        
        if "sample_packs" not in self.samples_data:
            return stats
        
        all_lengths = []
        
        for pack_id, pack_data in self.samples_data["sample_packs"].items():
            stats["total_packs"] += 1
            pack_samples = len(pack_data.get("samples", []))
            stats["total_samples"] += pack_samples
            
            pack_lengths = []
            for sample in pack_data.get("samples", []):
                text_len = len(sample.get("text", ""))
                all_lengths.append(text_len)
                pack_lengths.append(text_len)
            
            stats["pack_details"][pack_id] = {
                "name": pack_data.get("name", pack_id),
                "sample_count": pack_samples,
                "avg_text_length": sum(pack_lengths) / len(pack_lengths) if pack_lengths else 0
            }
        
        if all_lengths:
            stats["text_length_stats"]["min"] = min(all_lengths)
            stats["text_length_stats"]["max"] = max(all_lengths)
            stats["text_length_stats"]["avg"] = sum(all_lengths) / len(all_lengths)
        
        if stats["total_packs"] > 0:
            stats["avg_samples_per_pack"] = stats["total_samples"] / stats["total_packs"]
        
        return stats
    
    def generate_usage_report(self) -> str:
        """Generate a formatted report about the sample packs"""
        stats = self.get_sample_statistics()
        issues = self.validate_samples()
        
        report = "=== SAMPLE PACK REPORT ===\n\n"
        
        # Overview
        report += f"📊 OVERVIEW:\n"
        report += f"  - Total Packs: {stats['total_packs']}\n"
        report += f"  - Total Samples: {stats['total_samples']}\n"
        report += f"  - Average Samples per Pack: {stats['avg_samples_per_pack']:.1f}\n\n"
        
        # Text Statistics
        report += f"📝 TEXT STATISTICS:\n"
        report += f"  - Shortest Text: {stats['text_length_stats']['min']} characters\n"
        report += f"  - Longest Text: {stats['text_length_stats']['max']} characters\n"
        report += f"  - Average Length: {stats['text_length_stats']['avg']:.0f} characters\n\n"
        
        # Pack Details
        report += f"📦 PACK DETAILS:\n"
        for pack_id, details in stats['pack_details'].items():
            report += f"  - {details['name']}: {details['sample_count']} samples, "
            report += f"avg {details['avg_text_length']:.0f} chars\n"
        report += "\n"
        
        # Validation Issues
        if issues:
            report += f"⚠️  VALIDATION ISSUES:\n"
            for issue in issues:
                report += f"  - {issue}\n"
        else:
            report += f"✅ VALIDATION: All samples passed validation!\n"
        
        return report

def main():
    """Main function for command-line usage"""
    generator = SamplePackGenerator()
    
    print("Sample Pack Generator and Validator")
    print("==================================")
    
    # Generate report
    report = generator.generate_usage_report()
    print(report)
    
    # Validate samples
    issues = generator.validate_samples()
    if issues:
        print(f"\n❌ Found {len(issues)} validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All sample packs are valid!")

if __name__ == "__main__":
    main() 