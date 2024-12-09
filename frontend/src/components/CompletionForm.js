import React, { useState } from "react";
import axios from "axios";

const CompletionForm = () => {
  const [formData, setFormData] = useState({
    firstTermGpa: "",
    secondTermGpa: "",
    firstLanguage: "",
    funding: "",
    school: "",
    fastTrack: "",
    coop: "",
    residency: "",
    gender: "",
    previousEducation: "",
    ageGroup: "",
    highSchoolAverageMark: "",
    mathScore: "",
    englishScore: ""
  });

  const [completionResult, setCompletionResult] = useState(null);
  const [gpaResult, setGpaResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Request for program completion prediction
      const completionResponse = await axios.post("http://127.0.0.1:5000/predict/program-completion", formData);
      setCompletionResult(completionResponse.data.program_completion_prediction);
      
      // Request for GPA prediction
      const gpaResponse = await axios.post("http://127.0.0.1:5000/predict/gpa", formData);
      setGpaResult(gpaResponse.data.gpa_prediction);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-8 bg-white shadow-lg rounded-lg">
      <h2 className="text-3xl font-semibold text-center mb-6">Program Completion and GPA Prediction</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Numeric Fields with Limits */}
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">First Term GPA</label>
          <input
            type="number"
            name="firstTermGpa"
            value={formData.firstTermGpa}
            min="0"
            max="4.5"
            step="0.1"
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Second Term GPA</label>
          <input
            type="number"
            name="secondTermGpa"
            value={formData.secondTermGpa}
            min="0"
            max="4.5"
            step="0.1"
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">High School Average Mark</label>
          <input
            type="number"
            name="highSchoolAverageMark"
            value={formData.highSchoolAverageMark}
            min="0"
            max="100"
            step="0.1"
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Math Score</label>
          <input
            type="number"
            name="mathScore"
            value={formData.mathScore}
            min="0"
            max="50"
            step="0.1"
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          />
        </div>

        {/* Dropdown Options for Categorical Fields */}
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">First Language</label>
          <select
            name="firstLanguage"
            value={formData.firstLanguage}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">English</option>
            <option value="2">French</option>
            <option value="3">Other</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Funding</label>
          <select
            name="funding"
            value={formData.funding}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Apprentice_PS</option>
            <option value="2">GPOG_FT</option>
            <option value="3">Intl Offshore</option>
            <option value="4">Intl Regular</option>
            <option value="5">Intl Transfer</option>
            <option value="6">Joint Program Ryerson</option>
            <option value="7">Joint Program UTSC</option>
            <option value="8">Second Career Program</option>
            <option value="9">Work Safety Insurance Board</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">School</label>
          <select
            name="school"
            value={formData.school}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Advancement</option>
            <option value="2">Business</option>
            <option value="3">Communications</option>
            <option value="4">Community and Health</option>
            <option value="5">Hospitality</option>
            <option value="6">Engineering</option>
            <option value="7">Transportation</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Fast Track</label>
          <select
            name="fastTrack"
            value={formData.fastTrack}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Y</option>
            <option value="2">N</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Co-op</label>
          <select
            name="coop"
            value={formData.coop}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Y</option>
            <option value="2">N</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Residency</label>
          <select
            name="residency"
            value={formData.residency}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Domestic</option>
            <option value="2">International</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Gender</label>
          <select
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Female</option>
            <option value="2">Male</option>
            <option value="3">Neutral</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Previous Education</label>
          <select
            name="previousEducation"
            value={formData.previousEducation}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">HighSchool</option>
            <option value="2">PostSecondary</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">Age Group</label>
          <select
            name="ageGroup"
            value={formData.ageGroup}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">0 to 18</option>
            <option value="2">19 to 20</option>
            <option value="3">21 to 25</option>
            <option value="4">26 to 30</option>
            <option value="5">31 to 35</option>
            <option value="6">36 to 40</option>
            <option value="7">41 to 50</option>
            <option value="8">51 to 60</option>
            <option value="9">61 to 65</option>
            <option value="10">66+</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-lg font-medium mb-2">English Score</label>
          <select
            name="englishScore"
            value={formData.englishScore}
            onChange={handleChange}
            className="p-3 border border-gray-300 rounded-md"
          >
            <option value="">Select</option>
            <option value="1">Level-130</option>
            <option value="2">Level-131</option>
            <option value="3">Level-140</option>
            <option value="4">Level-141</option>
            <option value="5">Level-150</option>
            <option value="6">Level-151</option>
            <option value="7">Level-160</option>
            <option value="8">Level-161</option>
            <option value="9">Level-170</option>
            <option value="10">Level-171</option>
            <option value="11">Level-180</option>
          </select>
        </div>
        
        <button
          type="submit"
          className="w-full p-3 mt-6 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-300"
        >
          Predict
        </button>
      </form>
      {completionResult && <p className="mt-4 text-xl">Predicted Completion: {completionResult}</p>}
      {gpaResult && <p className="mt-4 text-xl">Predicted GPA: {gpaResult}</p>}
    </div>
  );
};

export default CompletionForm;
